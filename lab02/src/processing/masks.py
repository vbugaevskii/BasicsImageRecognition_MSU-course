#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import *
from utils.transform import *
from utils.morphology import *

from skimage.morphology import skeletonize


def create_hand_mask(img_gray):
    _, img_mask = cv2.threshold(img_gray, 45, 255, cv2.THRESH_BINARY)
    img_mask = scale_image_pixels(img_mask)
    img_mask = remove_holes(img_mask)

    skeleton = skeletonize(img_mask > 0)
    skeleton = scale_image_pixels(skeleton)
    skeleton = cv2.dilate(skeleton, brush_5)
    skeleton = skeleton > 0

    distance = cv2.distanceTransform(img_mask, cv2.DIST_L2, 5)

    r = distance[np.where(skeleton)]
    mask = np.logical_and(skeleton, distance >= .9 * r.max())
    centers = np.where(mask)
    radiuses = distance[centers] + 10

    img_mask_copy = img_mask.copy()

    center = np.unravel_index(np.argmax(distance, axis=None), distance.shape)
    radius_ = int(np.ceil(distance[center[0], center[1]]) + 10)

    img_mask = cv2.circle(img_mask, (center[1], center[0]), radius_, 0, -1)
    for row, col, radius in zip(centers[0], centers[1], radiuses):
        img_mask = cv2.circle(img_mask, (col, row), radius, 0, -1)
    img_mask = remove_border_objects(img_mask)

    # img_mask = remove_sparse_objects(img_mask, area=5000, solidity=0.5)

    img_mask = remove_small_objects(img_mask)

    img_mask = cv2.circle(img_mask, (center[1], center[0]), radius_, 255, -1)
    for row, col, radius in zip(centers[0], centers[1], radiuses):
        img_mask = cv2.circle(img_mask, (col, row), radius, 255, -1)
    img_mask = cv2.bitwise_and(img_mask, img_mask_copy)

    img_mask = cv2.bitwise_and(img_gray, img_gray, mask=img_mask)

    _, img_mask = cv2.threshold(img_mask, 55, 255, cv2.THRESH_BINARY)

    img_mask = find_largest_object(img_mask)
    img_mask = remove_holes(img_mask)

    return img_mask


def position_hand_mask(img_hand):
    tforms = []

    img_mask = img_hand.copy()

    img_labels = label(img_mask)
    img_props = regionprops(img_labels)[0]

    img_mask, tform = center_object_on_image(img_mask, img_props)
    tforms.append(tform)

    # Доворот, чтобы средний палец смотрел вверх
    angle = np.pi / 2 - img_props.orientation
    angle += np.pi / 15

    img_mask, tform = rotate_image(img_mask, angle)
    img_mask = find_largest_object(img_mask)
    tforms.append(tform)

    img_labels = label(img_mask)
    img_props = regionprops(img_labels)[0]
    centroid = img_props.centroid
    if centroid[0] < img_mask.shape[0] // 2:
        img_mask, tform = rotate_image(img_mask, np.pi)
        tforms.append(tform)

    img_mask = scale_image_pixels(img_mask)
    img_mask = cv2.erode(img_mask, brush_3, iterations=2)
    img_mask = cv2.dilate(img_mask, brush_3, iterations=2)
    img_mask = remove_holes(img_mask)

    img_mask = cv2.erode(img_mask, brush_5)
    for i in range(15):
        img_mask = cv2.medianBlur(img_mask, ksize=5)
    # img_mask = cv2.dilate(img_mask, brush_5)

    tform = reduce(lambda x, y: x + y, tforms[::-1])
    return img_mask, tform


def create_palm_and_fingers_masks(img_hand):
    img_mask = img_hand.copy()

    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, brush_70)

    img_fingers = img_hand - img_mask
    img_fingers = remove_small_objects(img_fingers, area=1200)

    img_fingers = filter_ellipse_objects(img_fingers)

    img_hand = np.maximum(img_mask, img_fingers)
    img_palm = img_mask.copy()

    return img_hand, img_palm, img_fingers
