#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import *

from skimage.feature import canny
from skimage.measure import label, regionprops

from scipy.signal import argrelmax


__all__ = ['features', 'utils']


def extract_tips_and_valleys(img_hand, img_palm, img_fingures):
    # Поиск кончиков пальцев
    img_contour = canny(img_hand)
    img_contour = scale_image_pixels(img_contour)

    img_props = regionprops(label(img_palm))[0]
    row_center, col_center = img_props.centroid
    center = np.array((col_center, row_center), ndmin=2)

    contour_p = np.where(img_contour > 0)
    contour_p = np.array(zip(contour_p[1], contour_p[0]))
    index = argsort_vertices(contour_p, center[0])
    contour_p = contour_p[index]
    contour_d = distance_matrix(center, contour_p)[0]

    contour_sup = contour_p[argrelmax(contour_d, order=70)[0]]
    index = argsort_vertices(contour_sup, center[0])
    contour_sup = contour_sup[index][:5]

    p_tips = contour_sup.copy()

    # Поиск впадин между пальцами
    _, contours, hierarchy = cv2.findContours(img_hand, 2, 1)
    contour = contours[0]

    convex_hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, convex_hull)

    p_valleys = []

    for i in range(defects.shape[0]):
        _, _, f, d_area = defects[i, 0]
        if d_area > 10000:
            p_valleys.append(tuple(contour[f][0]))

    radius = int(img_props.major_axis_length) // 2 + 30
    p_valleys = filter(lambda p: np.linalg.norm(np.array(p) - center[0]) < radius, p_valleys)
    index = argsort_vertices(p_valleys, center[0])
    p_valleys = [p_valleys[i] for i in index]

    # Если мезинец сильно отогнут, то он тоже создает большую область
    if len(p_valleys) > 4:
        p_valleys = p_valleys[1:]

    # Объединяем в отдельный результат и применяем обратные преобразования
    assert len(p_tips) == 5 and len(p_valleys) == 4

    vertices = []
    for i in range(len(p_valleys)):
        vertices.append(p_tips[i])
        vertices.append(p_valleys[i])
    vertices.append(p_tips[i + 1])
    vertices = np.vstack(vertices)

    return vertices


def draw_vertices_on_image(img_color, vertices):
    vertices = vertices.astype(np.int32)
    img_color = img_color.copy()

    for i in range(len(vertices) - 1):
        col_1, row_1 = vertices[i]
        col_2, row_2 = vertices[i + 1]
        cv2.line(img_color, (col_1, row_1), (col_2, row_2), (34, 199, 34), 2)

    for i, (col, row) in enumerate(vertices):
        color = (199, 34, 34) if i % 2 else (34, 34, 199)
        cv2.circle(img_color, (col, row), 6, color, -1)
        cv2.circle(img_color, (col, row), 3, (255, 255, 255), -1)

    return img_color
