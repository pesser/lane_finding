import glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor


def calibrate_camera():
    """Calibrate camera from chessboard pictures."""
    # chessboard pattern parameters
    nx = 9
    ny = 6
    # chessboard pictures
    calibration_imgs = glob.glob("./camera_cal/*.jpg")

    # object coordinates of chessboard pattern in same order as corners
    # returned by findChessboardCorners, i.e. column first
    objcoords = np.zeros((ny*nx, 3), dtype = np.float32)
    for i in range(ny):
        for j in range(nx):
            objcoords[i * nx + j] = (i, j, 0)

    print("Calibrating with {} images.".format(len(calibration_imgs)))
    objpoints = list()
    imgpoints = list()
    for fname in calibration_imgs:
        # load image
        img = cv2.imread(fname)
        # convert to grayscale for chessboard detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect corners
        found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if not found:
            print("Warning: Could not find chessboard corners for {}".format(fname))
        else:
            objpoints.append(objcoords)
            imgpoints.append(corners)

    # image size is expected as (width, height)
    res = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
    cmatrix, cparams = res[1], res[2]
    
    return cmatrix, cparams


def visualize_distortion(params, in_fname, out_fname):
    """Visualize distortion correction."""
    img = cv2.imread(in_fname)
    cal = cv2.undistort(img, params[0], params[1])

    # convert from opencv's bgr to rgb for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cal = cv2.cvtColor(cal, cv2.COLOR_BGR2RGB)

    f, axes = plt.subplots(1, 2, figsize = (15,5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].set_axis_off()
    axes[1].imshow(cal)
    axes[1].set_title("Distortion Corrected")
    axes[1].set_axis_off()
    f.savefig(out_fname)


def get_warp_params():
    """Return transform into and out of birds-eye view."""
    c = 15 # correction for fine tuning
    src = np.float32([
        [593-c,460], [687+c,460],
        [190,720], [1090,720]])
    dst = np.float32([
        [391.5,0], [888.5,0],
        [391.5,720], [888.5,720]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return (M, M_inv)


def threshold(img):
    """Detect candidate points for lane detection based on gradients and
    colors."""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    thresholds = {
            "yellow": {
                "h_low": 15,
                "h_high": 41,
                "l_low": 0,
                "l_high": 256,
                "s_low": 50,
                "s_high": 256},
            "white": {
                "h_low": 0,
                "h_high": 256,
                "l_low": 180,
                "l_high": 256,
                "s_low": 0,
                "s_high": 256}}

    gradient_threshold = 20
    gradient_channels = [1,2] # hue gradient is not informative here
    sobel_kernel = 9

    # compute masks
    masks = dict()
    for color in thresholds:
        ths = thresholds[color]
        masks[color] = (
                (ths["h_low"] <= hls[:,:,0]) &
                (hls[:,:,0] < ths["h_high"]) &
                (ths["l_low"] <= hls[:,:,1]) &
                (hls[:,:,1] < ths["l_high"]) &
                (ths["s_low"] <= hls[:,:,2]) &
                (hls[:,:,2] < ths["s_high"]))

    # combine masks
    mask = np.any(np.stack(masks.values(), axis = -1), axis = -1)

    # compute gradients
    hls = np.float64(hls) / 255.0
    for c in gradient_channels:
        hls[:,:,c] = np.abs(cv2.Sobel(hls[:,:,c], cv2.CV_64F, 1, 0, ksize = sobel_kernel))
        hls[:,:,c] = hls[:,:,c] / np.max(hls[:,:,c])
    hls = np.uint8(255.0 * hls)

    # apply mask to gradients
    for c in range(3):
        if c in gradient_channels:
            hls[:,:,c][~mask] = 0
        else:
            hls[:,:,c] = 0
        hls[:,:,c][gradient_threshold <= hls[:,:,c]] = 255

    # combine all channels
    ths = np.max(hls, axis = -1)

    return ths


def get_lane_masks(img):
    """Take in a thresholded image in birds eye view and return a mask of
    likely lane positions."""
    bottom_half = img[img.shape[0]//2:, :]
    histogram = np.sum(bottom_half, axis = 0)

    # initial guess for lane position based on peaks in the histogram
    x_center = img.shape[1]//2
    x_left = np.argmax(histogram[:x_center])
    x_right = x_center + np.argmax(histogram[x_center:])

    n_windows = 8
    w_height = img.shape[0]//n_windows
    w_width = 80 # actually _half_ window width
    w_center = [x_left, x_right]

    window_n_matches = [None, None]
    window_offset = [None, None]
    min_matches = 50

    # left and right mask
    mask = np.zeros_like(img, dtype = bool)
    mask = np.repeat(mask[:,:,None], 2, axis = 2)

    # proceed from bottom up in windows
    for i in range(n_windows):
        # for both left and right lane
        for k in range(2):
            # current search window
            w_top = (n_windows - 1 - i) * w_height
            w_bottom = (n_windows - i) * w_height
            w_left = w_center[k] - w_width
            w_right = w_center[k] + w_width

            # add rectangle to mask
            mask[w_top:w_bottom, w_left:w_right, k] = True

            # estimate lane position within window
            window_values = img[w_top:w_bottom, w_left:w_right]
            window_nonzeros = np.nonzero(window_values)
            window_xs = window_nonzeros[1]
            window_n_matches[k] = window_xs.shape[0]
            if not window_n_matches[k] > 0:
                window_xs = [w_width]
            window_offset[k] = int(np.mean(window_xs) - w_width)

        # lines are assumed parallel thus we update the position of the
        # search window for both lines based on the more confident estimate
        if np.max(window_n_matches) > min_matches:
            best_k = np.argmax(window_n_matches)
            offset = window_offset[best_k]
            for k in range(2):
                w_center[k] = w_center[k] + offset

    return mask


def get_polyfit(img, mask):
    """Fit polynomial to lanes in a thresholded, birds-eye image using given
    mask for lanes."""
    # fit a polynomial to match points within mask
    fit = [None, None]
    for k in range(2):
        lane = np.array(img)
        lane[~mask[:,:,k]] = 0
        lane_points = np.nonzero(lane)
        if lane_points[0].shape[0] > 0:
            # fit polynomial with x and y swapped since lanes are expected to be
            # vertical
            fit[k] = np.polyfit(lane_points[0], lane_points[1], deg = 2)
        else:
            # no candidate points
            fit[k] = None

    return fit


def visualize_fitting(thresholded, mask, fit):
    """Visualize thresholding, search mask and final fit."""
    mask_img = np.concatenate([255*np.uint8(mask), np.zeros((mask.shape[0], mask.shape[1], 1), dtype = np.uint8)], axis = -1)
    thresholded = np.repeat(thresholded[:,:,None], 3, axis = 2)
    out_img = cv2.addWeighted(thresholded, 1.0, mask_img, 0.3, 0.0)
    return out_img


def lanes(img, fit):
    """Visualize polynomial fit to thresholded img using polynomial fit."""
    # visualize polynomial fit
    out_img = np.zeros_like(img)[:,:,None]
    out_img = np.repeat(out_img, 3, axis = 2)
    line_windows = [None, None]
    for k in range(2):
        ys = np.linspace(0, img.shape[0] - 1, 100)
        xs = np.polyval(fit[k], ys)
        margin = 10
        left_line_window = np.array([np.transpose(np.vstack([xs - margin, ys]))])
        right_line_window = np.array([np.flipud(np.transpose(np.vstack([xs + margin, ys])))])
        line_window = np.hstack([left_line_window, right_line_window])
        color = [0,0,0]
        color[k] = 255
        cv2.fillPoly(out_img, np.int_([line_window]), color)
        if k == 0:
            line_windows[k] = np.array([np.transpose(np.vstack([xs, ys]))])
        else:
            line_windows[k] = np.array([np.flipud(np.transpose(np.vstack([xs, ys])))])

    line_window = np.hstack(line_windows)
    color = [0,0,0]
    color[2] = 255
    cv2.fillPoly(out_img, np.int_([line_window]), color)

    return out_img


def backproject(orig, lanes, warp_params):
    """Project lanes from birds-eye perspective to camera perspective and
    combine with original image."""
    back = unwarp(lanes, warp_params)
    combined = cv2.addWeighted(orig, 1.0, back, 0.5, 0.0)
    return combined


def undistort(x, calibration_params):
    return cv2.undistort(x, calibration_params[0], calibration_params[1])


def warp(x, warp_params):
    return cv2.warpPerspective(x, warp_params[0], (x.shape[1], x.shape[0]), flags = cv2.INTER_LINEAR)


def unwarp(x, warp_params):
    return cv2.warpPerspective(x, warp_params[1], (x.shape[1], x.shape[0]), flags = cv2.INTER_LINEAR)


def mask_from_poly(x, poly):
    """Build mask around polynomial fit."""
    assert(len(x.shape) == 2)

    margin = 70

    # draw boundary around lane
    out_img = np.zeros_like(x)[:,:,None]
    out_img = np.repeat(out_img, 3, axis = 2)
    line_windows = [None, None]
    for k in range(2):
        ys = np.linspace(0, x.shape[0] - 1, 100)
        xs = np.polyval(poly[k], ys)
        left_line_window = np.array([np.transpose(np.vstack([xs - margin, ys]))])
        right_line_window = np.array([np.flipud(np.transpose(np.vstack([xs + margin, ys])))])
        line_window = np.hstack([left_line_window, right_line_window])
        color = [0,0,0]
        color[k] = 255
        cv2.fillPoly(out_img, np.int_([line_window]), color)

    # left and right mask
    mask = np.zeros_like(x, dtype = bool)
    mask = np.repeat(mask[:,:,None], 2, axis = 2)
    mask[:,:,0] = out_img[:,:,0] == 255
    mask[:,:,1] = out_img[:,:,1] == 255

    return mask


def infer_radii(y, fits, scales):
    """Infer radii at point y from polynomial fittings."""
    radii = [0,0]
    for i in range(len(fits)):
        fit = np.array(fits[i])
        # convert polynomial coefficients to scale
        # the formula follows from comparing coeffcients of the original
        # polynomial with f(y_j) = x_j with the scaled polynomial
        # g(scale[0]*y_j) = scale[1]*f(y_j)
        d = fit.shape[0] - 1
        for k in range(fit.shape[0]):
            fit[k] = scales[1] * fit[k] / (scales[0]**(d-k))
        radii[i] = (1.0 + (2*fit[0]*y*scales[0] + fit[1])**2)**(3/2) / abs(2*fit[0])

    return radii


def infer_offset(x, y, fits, scales):
    """Infer horizontal lane center offset from x at height y."""
    left_lane = np.polyval(fits[0], y)
    right_lane = np.polyval(fits[1], y)
    lane_center = 0.5 * (left_lane + right_lane)
    pixel_offset = lane_center - x
    scale_offset = scales[1] * pixel_offset
    
    return scale_offset


class SmoothLanes(object):
    """Lane detection pipeline which keeps track of its detections to
    improve performance."""

    def __init__(self, calibration_params, warp_params):
        self.calibration_params = calibration_params
        self.warp_params = warp_params

        # smoothness parameter for moving average of polynomial coefficients
        self.decay = 0.25
        # outlier tolerance on squared l2 distance between new and previous
        # polynomial coefficients
        self.TOL = 1e+4
        # scales to convert to radius in meters
        self.scales = [30/720, 3.7/700]

        self.prev_polyfit = None


    def __call__(self, x):
        # this class is used with moviepy which uses rgb ordering but the
        # pipeline was developed with opencv which uses bgr ordering thus we
        # have to convert
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

        x = undistort(x, calibration_params)
        orig = x # keep it for visualization
        x = warp(x, self.warp_params)
        x = threshold(x)
        thresholded = x

        if self.prev_polyfit:
            # reuse previous detection for mask
            mask = mask_from_poly(thresholded, self.prev_polyfit)
        else:
            # build mask from scratch
            mask = get_lane_masks(thresholded)

        # fit polynomial using the mask
        polyfit = get_polyfit(thresholded, mask)

        if polyfit[0] is not None and polyfit[1] is not None:
            # both lanes could be fitted - all good
            detected = "both"
            pass
        elif polyfit[0] is None and polyfit[1] is None:
            # neither lane could be fitted, reuse previous fit
            detected = "none"
            polyfit = self.prev_polyfit
        else:
            # one of the lanes could be fitted - use curvature of detected
            # lane at last known offset
            detected = "one"
            lost_lane = polyfit.index(None)
            found_lane = 1 - lost_lane
            polyfit[lost_lane] = np.array(polyfit[found_lane])
            bottom_y = x.shape[0] - 1
            bottom_offset = np.polyval(self.prev_polyfit[lost_lane], bottom_y) - np.polyval(self.prev_polyfit[found_lane], bottom_y)
            polyfit[lost_lane][-1] += bottom_offset

        diff = [0.0,0.0]
        if self.prev_polyfit is not None:
            for k in range(2):
                diff[k] = np.sum(np.square(polyfit[k] - self.prev_polyfit[k]))
                if diff[k] < self.TOL:
                    # update exponential moving average of parameters
                    for i in range(len(polyfit[k])):
                        polyfit[k][i] = (
                                self.decay * polyfit[k][i] +
                                (1.0 - self.decay) * self.prev_polyfit[k][i])
                    self.prev_polyfit[k] = polyfit[k]
                else:
                    # use previous
                    polyfit = self.prev_polyfit
        else:
            # initialize
            self.prev_polyfit = polyfit

        x = lanes(thresholded, polyfit)
        x = backproject(orig, x, self.warp_params)

        # overlay info
        radii = infer_radii(x.shape[0] - 1, polyfit, self.scales)
        center_offset = infer_offset(x.shape[1] // 2, x.shape[0] - 1 - 20, polyfit, self.scales)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = " | ".join([
            "detected: " + detected,
            "diff: {}".format(", ".join(["{:.4e}".format(x) for x in diff])),
            "offset: {:.3f} [m]".format(center_offset),
            "radii: {} [m]".format(", ".join(["{:.1f}".format(x) for x in radii]))])
        x = cv2.putText(x, txt, (10, x.shape[0] - 1 - 10), font, 0.8, (255,0,0))

        # convert back to rgb again
        x = cv2.cvtColor(np.uint8(x), cv2.COLOR_BGR2RGB)
        return x


if __name__ == "__main__":
    # calibration
    calibration_params = calibrate_camera()
    # warp parameters to transform to birds eye view
    warp_params = get_warp_params()

    visualize_distortion(
            calibration_params,
            "./camera_cal/calibration5.jpg",
            "./output_images/calibration.png")
    visualize_distortion(
            calibration_params,
            "./test_images/test4.jpg",
            "./output_images/test_calibration.png")

    # evaluate on test images
    for fname in glob.glob("./test_images/*.jpg"):
        img = cv2.imread(fname)

        undistorted = undistort(img, calibration_params)
        out_fname = "./output_images/undistorted_{}".format(os.path.basename(fname))
        cv2.imwrite(out_fname, undistorted)

        warped = warp(undistorted, warp_params)
        out_fname = "./output_images/warped_{}".format(os.path.basename(fname))
        cv2.imwrite(out_fname, warped)

        thresholded = threshold(warped)
        out_fname = "./output_images/thresholded_{}".format(os.path.basename(fname))
        cv2.imwrite(out_fname, thresholded)

        mask = get_lane_masks(thresholded)
        polyfit = get_polyfit(thresholded, mask)

        fitting = visualize_fitting(thresholded, mask, polyfit)
        lanes_img = lanes(thresholded, polyfit)

        fitting_vis = cv2.addWeighted(fitting, 0.5, lanes_img, 0.5, 0.0)
        out_fname = "./output_images/fitting_{}".format(os.path.basename(fname))
        cv2.imwrite(out_fname, fitting_vis)

        lanes_vis = cv2.addWeighted(warped, 1.0, lanes_img, 0.5, 0.0)
        out_fname = "./output_images/lanes_{}".format(os.path.basename(fname))
        cv2.imwrite(out_fname, lanes_vis)

        backprojected = backproject(undistorted, lanes_img, warp_params)
        out_fname = "./output_images/backprojected_{}".format(os.path.basename(fname))
        cv2.imwrite(out_fname, backprojected)

    # evaluate on video
    for clip_fname in ["project_video.mp4", "challenge_video.mp4"]:
        smooth_lanes = SmoothLanes(calibration_params, warp_params)
        out_clip_fname = "out_" + clip_fname
        clip = moviepy.editor.VideoFileClip(clip_fname)
        out_clip = clip.fl_image(smooth_lanes)
        out_clip.write_videofile(out_clip_fname, audio = False)
