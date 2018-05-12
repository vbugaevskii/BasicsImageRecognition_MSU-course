#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import cv2

from scipy.spatial import distance_matrix


def create_brush(d):
    brush = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(brush, (d // 2, d // 2), d // 2, 1, -1)
    return brush


brush_3 = np.ones((3, 3), dtype=np.uint8)
brush_3[0,0] = brush_3[0,2] = brush_3[2,0] = brush_3[2,2] = 0

brush_5 = np.ones((5, 5), dtype=np.uint8)
brush_5[0,0] = brush_5[0,4] = brush_5[4,0] = brush_5[4,4] = 0

brush_15 = create_brush(15)

brush_70 = create_brush(70)


def check_image_consistency(img):
    return img.min() == 0 and img.max() == 255 and img.dtype == np.uint8


def scale_image_pixels(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)


def argsort_vertices(vertices, center):
    vertices = np.array(vertices, dtype=np.float32, ndmin=2)
    center = np.array(center, dtype=np.float32, ndmin=2)

    assert 2 == vertices.shape[1] == center.shape[1]

    def cart2pol(vectors):
        rho = np.linalg.norm(vectors, axis=1)
        phi = np.arctan2(vectors[:, 1], vectors[:, 0])
        return rho, phi

    vertices -= center
    vertices *= np.array([[1, -1]])
    _, angles = cart2pol(vertices)

    index = np.argsort(angles)[::-1]
    return index


def choose_unique_points(points, thrsh=15.0):
    points = np.array(points)

    points_dist = distance_matrix(points, points) < thrsh
    points_order = np.argsort(points_dist.sum(axis=0))[::-1]
    points = points[points_order]
    points_dist = points_dist[points_order][:, points_order]

    points_processed, points_chosen = set(), set()

    for i in range(len(points)):
        if i in points_processed:
            continue
        points_processed.add(i)
        points_chosen.add(i)
        points_processed |= set(np.where(points_dist[i])[0].tolist())

    points = points[list(points_chosen)]
    return points


def padding(img, new_shape):
    assert 2 <= len(img.shape) <= 3

    img_r, img_c = img.shape[:2]

    r_padding = (new_shape[0] - img_r) // 2
    c_padding = (new_shape[1] - img_c) // 2

    m_padding = (
        (r_padding, new_shape[0] - (img_r + r_padding)),
        (c_padding, new_shape[1] - (img_c + c_padding)),
    )

    if len(img.shape) == 3:
        m_padding = (m_padding[0], m_padding[1], (0, 0))

    img = np.pad(img, m_padding, 'constant')
    return img
