#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from skimage.filters import sobel

from src.utils.morphology import remove_holes, remove_appendix_objects
from src.utils import brush_3, brush_5, scale_image_pixels


def func(x):
    mask = x < 96
    y = np.zeros(x.shape)
    y[mask] = x[mask] / 4.0
    y[~mask] = 1.45 * x[~mask] - 115.2
    return y


def create_mask_figures(img_gray, mask):
    proc_pb = sobel(img_gray, mask=mask)
    proc_pb = np.abs(proc_pb)

    proc_pb = scale_image_pixels(proc_pb)

    proc_pb = cv2.medianBlur(proc_pb, ksize=5)
    proc_pb = cv2.medianBlur(proc_pb, ksize=5)
    proc_pb = cv2.medianBlur(proc_pb, ksize=5)

    proc_pb = scale_image_pixels(proc_pb)
    proc_pb = proc_pb.astype(np.uint8)

    proc_pb = cv2.equalizeHist(proc_pb)
    proc_pb = func(proc_pb)
    proc_pb = func(proc_pb)
    proc_pb = proc_pb.astype(np.uint8)

    q = np.percentile(filter(lambda x: x > 0, proc_pb.reshape(-1)), 75)
    _, proc_pb = cv2.threshold(proc_pb.astype(np.uint8), q, 255, cv2.THRESH_BINARY)
    proc_pb = remove_holes(proc_pb)

    proc_pb = cv2.morphologyEx(proc_pb, cv2.MORPH_OPEN, brush_5, iterations=2)
    proc_pb = remove_appendix_objects(proc_pb, area=360, perimeter=160)
    proc_pb = cv2.dilate(proc_pb, brush_3, iterations=4)

    return proc_pb
