#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import watershed

from scipy import ndimage as ndi


def remove_holes(img):
    img_flood_fill = img.copy()

    h, w = img_flood_fill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(img_flood_fill, mask, (0, 0), 255)
    img_flood_fill = cv2.bitwise_not(img_flood_fill)

    return img | img_flood_fill


def remove_small_objects(img, img_labels, area=300, perimeter=140):
    img_proc = img.copy()
    img_labels_remove = set([prop.label for prop in regionprops(img_labels)
                             if prop.perimeter < perimeter or prop.area < area])

    mask = map(lambda x: x in img_labels_remove, img_labels.reshape(-1))
    mask = np.array(mask).reshape(img_labels.shape)

    img_proc[mask] = 0
    return img_proc


def find_borders(labels):
    res = np.zeros(labels.shape)

    for j in range(labels.shape[1]):
        if j < labels.shape[1] - 1:
            mask = np.logical_and(labels[:, j] > 0, labels[:, j + 1] > 0)
            res[mask, j] = np.logical_or(res[mask, j],
                                         (labels[mask, j] - labels[mask, j + 1]) != 0)

        if j > 0:
            mask = np.logical_and(labels[:, j] > 0, labels[:, j - 1] > 0)
            res[mask, j] = np.logical_or(res[mask, j],
                                         (labels[mask, j] - labels[mask, j - 1]) != 0)

    for i in range(labels.shape[0]):
        if i < labels.shape[0] - 1:
            mask = np.logical_and(labels[i, :] > 0, labels[i + 1, :] > 0)
            res[i, mask] = np.logical_or(
                res[i, mask], (labels[i, mask] - labels[i + 1, mask]) != 0)

        if i > 0:
            mask = np.logical_and(labels[i, :] > 0, labels[i - 1, :] > 0)
            res[i, mask] = np.logical_or(
                res[i, mask], (labels[i, mask] - labels[i - 1, mask]) != 0)

    res *= 255
    res = cv2.dilate(res, np.ones((2, 2)), iterations=2)
    return res


def remove_borders(borders, labels):
    labels_ = labels.copy()

    borders_m = label(borders)
    borders_remove = set([prop.label for prop in regionprops(borders_m)
                          if prop.perimeter < 55])

    mask = np.array(map(lambda x: x in borders_remove, borders_m.reshape(-1))). \
        reshape(borders_m.shape)
    borders[mask] = 0

    borders_m = label(borders)
    for i in range(1, borders_m.max() + 1):
        uniq = filter(lambda x: x > 0, np.unique(labels_[borders_m == i]))
        for i in uniq:
            labels_[labels == i] = min(uniq)

    return labels_


def remove_appendix_objects(img, area=300, perimeter=140):
    proc_pb = img.copy()

    distance = ndi.distance_transform_edt(proc_pb)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((17, 17)),
                                labels=proc_pb, exclude_border=4)
    markers = ndi.label(local_maxi)[0]
    img_labels = watershed(-distance, markers, mask=proc_pb)

    borders = find_borders(img_labels)
    img_labels = remove_borders(borders, img_labels)

    proc_pb = remove_small_objects(proc_pb, img_labels, area, perimeter)

    img_labels = label(proc_pb)
    proc_pb = remove_small_objects(proc_pb, img_labels, area, perimeter)

    return proc_pb

