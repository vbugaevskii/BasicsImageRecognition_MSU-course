#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from skimage.measure import label, regionprops

from src.utils.morphology import remove_holes
from src.utils import brush_3


def create_mask_cards(img_color):
    img_proc = img_color.copy()

    mask = img_proc[:, :, 0] > img_proc[:, :, 1:].sum(axis=2) / 1.5
    img_proc[np.where(mask)] = 0
    img_proc[np.where(~mask)] = 255
    img_proc = img_proc[:, :, 0]
    img_proc = cv2.bitwise_not(img_proc)

    img_proc = cv2.dilate(img_proc, kernel=brush_3, iterations=2)
    img_proc = cv2.erode(img_proc, kernel=brush_3, iterations=2)

    img_proc = remove_holes(img_proc)
    img_proc = cv2.medianBlur(img_proc, ksize=5)

    img_proc = remove_holes(img_proc)
    img_proc = cv2.erode(img_proc, kernel=brush_3)

    img_labels = label(img_proc)
    img_labels_remove = set([prop.label for prop in regionprops(img_labels) if prop.area < 3000])

    mask = map(lambda x: x in img_labels_remove, img_labels.reshape(-1))
    mask = np.array(mask).reshape(img_labels.shape)
    img_proc[mask] = 0

    return img_proc
