#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import check_image_consistency, scale_image_pixels

import numpy as np

from skimage import transform as tf


def center_object_on_image(img, img_props):
    assert check_image_consistency(img)

    min_row, min_col, _, _ = img_props.bbox
    tform1 = tf.SimilarityTransform(translation=(min_col, min_row))

    box_row_num, box_col_num = img_props.filled_image.shape
    img_row_num, img_col_num = img.shape[0], img.shape[1]

    row = (img_row_num - box_row_num) // 2
    col = (img_col_num - box_col_num) // 2

    tform2 = tf.SimilarityTransform(translation=(-col, -row))
    tform = tform2 + tform1

    img = tf.warp(img, tform)
    img = scale_image_pixels(img)
    img = (img > 128).astype(np.uint8) * 255

    assert check_image_consistency(img)
    return img, tform


def rotate_image(img, rads):
    assert check_image_consistency(img)

    rows, cols = img.shape[0], img.shape[1]
    center = np.array((cols, rows)) / 2. - 0.5

    tform1 = tf.SimilarityTransform(translation=center)
    tform2 = tf.SimilarityTransform(rotation=rads)
    tform3 = tf.SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1

    img = tf.warp(img, tform)
    img = scale_image_pixels(img)
    img = (img > 200).astype(np.uint8) * 255

    assert check_image_consistency(img)
    return img, tform


def transform_vertices(vertices, tform):
    assert vertices.shape[1] == 2

    vertices = np.hstack((vertices, np.ones(shape=(vertices.shape[0], 1))))
    vertices = np.matmul(tform.params, vertices.T).T
    vertices = vertices[:, :2]

    return vertices
