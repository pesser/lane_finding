#!/usr/bin/env python

"""Interactive Exploration of parameters for Line Detection."""

import numpy as np
import cv2

TOL = 1e-6
N_FLAGS = 3 # number of flags that control image generation, see make_img


def make_img_best(img):
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

    gradient_channels = [1,2]
    sobel_kernel = 9

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

    mask = np.any(np.stack(masks.values(), axis = -1), axis = -1)

    hls = np.float64(hls) / 255.0
    for c in gradient_channels:
        hls[:,:,c] = np.abs(cv2.Sobel(hls[:,:,c], cv2.CV_64F, 1, 0, ksize = sobel_kernel))
        hls[:,:,c] = hls[:,:,c] / np.max(hls[:,:,c])
    hls = np.uint8(255.0 * hls)

    for c in range(3):
        if c in gradient_channels:
            hls[:,:,c][~mask] = 0
        else:
            hls[:,:,c] = 0
        hls[:,:,c][20 <= hls[:,:,c]] = 255

    ths = np.max(hls, axis = -1)

    return ths



def make_img(img, mode, **kwargs):
    """Make line image for the given img."""

    # use fixed parameters in mode 0
    # otherwise mode represent a mask for the three channels
    if not mode:
        return make_img_best(img)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    mask = (
            (kwargs["h_low"] <= hls[:,:,0]) &
            (hls[:,:,0] < kwargs["h_high"]) &
            (kwargs["l_low"] <= hls[:,:,1]) &
            (hls[:,:,1] < kwargs["l_high"]) &
            (kwargs["s_low"] <= hls[:,:,2]) &
            (hls[:,:,2] < kwargs["s_high"]))

    if kwargs["grad"]:
        hls = np.float64(hls) / 255.0
        for c in range(3):
            hls[:,:,c] = np.abs(cv2.Sobel(hls[:,:,c], cv2.CV_64F, 1, 0, ksize = 9))
            hls[:,:,c] = hls[:,:,c] / np.max(hls[:,:,c])
        hls = np.uint8(255.0 * hls)

    for c in range(3):
        hls[:,:,c][~mask] = 0
        if not mode & 1<<c:
            hls[:,:,c] = 0

    return hls


if __name__ == "__main__":
    import sys
    import itertools

    if len(sys.argv) < 2:
        print("Useage: {} input_image.jpg [input_image2.jpg ...]".format(sys.argv[0]))
        print("Cycle through images with 'n'. Exit with 'q'.")
        exit(1)

    fnames = sys.argv[1:]

    # read images
    images = []
    for fname in fnames:
        images.append(cv2.imread(fname))
    print("Read {} images.".format(len(images)))

    # Line finder
    image_cycle = itertools.cycle(images)
    image = next(image_cycle)
    state = {
            "img": image,
            "result": None,
            "h_low": 0,
            "h_high": 256,
            "l_low": 0,
            "l_high": 256,
            "s_low": 0,
            "s_high": 256,
            "grad": 0,
            "mode": int("111", 2)}

    # parameters exposed to ui together with maximum values
    ui_max_params = {
            "h_low": 256,
            "h_high": 256,
            "l_low": 256,
            "l_high": 256,
            "s_low": 256,
            "s_high": 256,
            "grad": 1,
            "mode": (1 << N_FLAGS) - 1}

    # Create window to show result and controls
    cv2.namedWindow("Line Detection", 0)

    def update(param, k, state):
        state[k] = param
        state["result"] = None

    for k in sorted(ui_max_params):
        cv2.createTrackbar(
                k, "Line Detection",
                state[k],
                ui_max_params[k],
                lambda param, k = k, state = state: update(param, k, state))

    # Main loop
    while True:
        if state["result"] is None:
            state["result"] = make_img(**state)

        cv2.imshow("Line Detection", state["result"])

        key = cv2.waitKey(300)
        if  key == ord('q'):
            # quit
            break
        elif key == ord('n'):
            # next image
            image = next(image_cycle)
            update(image, "img", state)
