#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import check_image_consistency, scale_image_pixels

import cv2
import numpy as np

from operator import attrgetter

from skimage.measure import label, regionprops

from scipy.ndimage.morphology import binary_fill_holes


def remove_holes(img):
    assert check_image_consistency(img)

    img = (img > 0).astype(np.uint8)
    img = binary_fill_holes(img)
    img = scale_image_pixels(img)

    assert check_image_consistency(img)
    return img


def remove_border_objects(img):
    assert check_image_consistency(img)

    img = img.copy()
    img[0, :] = 255
    img[:, 0] = 255
    img[img.shape[0] - 1, :] = 255
    img[:, img.shape[1] - 1] = 255

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(img, mask, (0, 0), 0)

    assert check_image_consistency(img)
    return img


def remove_sparse_objects(img, area=300, solidity=0.9):
    assert check_image_consistency(img)

    img = img.copy()
    img_labels = label(img)

    img_labels_remove = set([prop.label for prop in regionprops(img_labels)
                             if prop.area < area and prop.solidity < solidity])

    mask = map(lambda x: x in img_labels_remove, img_labels.reshape(-1))
    mask = np.array(mask).reshape(img_labels.shape)

    img[mask] = 0

    assert check_image_consistency(img)
    return img


def remove_small_objects(img, area=300):
    assert check_image_consistency(img)

    img = img.copy()
    img_labels = label(img)

    img_labels_remove = set([prop.label for prop in regionprops(img_labels)
                             if prop.area < area])

    mask = map(lambda x: x in img_labels_remove, img_labels.reshape(-1))
    mask = np.array(mask).reshape(img_labels.shape)

    img[mask] = 0

    assert check_image_consistency(img)
    return img


def find_largest_object(img, copy=True):
    assert check_image_consistency(img)

    if copy:
        img = img.copy()

    img_labels = label(img)
    img_props = sorted(regionprops(img_labels), key=attrgetter('area'), reverse=True)[0]
    img[np.where(img_labels != img_props.label)] = 0

    assert check_image_consistency(img)
    return img


def filter_ellipse_objects(img):
    assert check_image_consistency(img)

    img = img.copy()
    img_labels = label(img)

    img_labels_remove = set([prop.label for prop in regionprops(img_labels)
                             if prop.major_axis_length == 0 or
                             prop.minor_axis_length / prop.major_axis_length > 0.6])

    mask = map(lambda x: x in img_labels_remove, img_labels.reshape(-1))
    mask = np.array(mask).reshape(img_labels.shape)

    img[mask] = 0

    assert check_image_consistency(img)
    return img


def skeleton_endpoints(skeleton, neighbours=1):
    skeleton = skeleton.copy()
    skeleton[skeleton > 0] = 1
    skeleton = skeleton.astype(np.uint8)

    kernel = np.array(
        [[1,  1, 1],
         [1, 10, 1],
         [1,  1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton, -1, kernel)

    result = np.zeros(skeleton.shape)
    result[np.where(filtered == 10 + neighbours)] = 1
    return result

