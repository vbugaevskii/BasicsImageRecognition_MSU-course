#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimage.measure import label, regionprops

import numpy as np

brush_3 = np.ones((3, 3), dtype=np.uint8)
brush_3[0, 0] = brush_3[0, 2] = brush_3[2, 0] = brush_3[2, 2] = 0

brush_5 = np.ones((5, 5), dtype=np.uint8)
brush_5[0, 0] = brush_5[0, 4] = brush_5[4, 0] = brush_5[4, 4] = 0


def scale_image_pixels(img):
    img = 255.0 * (img - img.min()).astype(float) / (img.max() - img.min())
    return img.astype(np.uint8)


def make_curbing_image(proc_c, size=(128, 128)):
    proc = np.zeros(size, dtype=np.uint8)

    props_raw = regionprops(label(proc_c))[0]
    y_cent, x_cent = map(lambda x: x // 2, props_raw.image.shape)
    y_size, x_size = props_raw.image.shape
    y_min, x_min = map(int, [proc.shape[0] // 2 - y_cent, proc.shape[1] // 2 - x_cent])

    proc[y_min:y_min + y_size, x_min:x_min + x_size] = props_raw.image * 255
    return proc
