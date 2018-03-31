#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pandas as pd

from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.feature import canny

from src.utils import scale_image_pixels, brush_3
from src.utils.morphology import remove_holes, remove_appendix_objects

from tangent import filter_hough_lines
from convexity import *

from scipy.spatial.distance import euclidean


def get_min_max_distance(proc_c):
    edges = canny(proc_c, sigma=0.1)
    edges = np.where(edges > 0)

    props_raw = regionprops(label(proc_c))[0]
    d_max, d_min = 0, np.inf

    for p in zip(edges[0], edges[1]):
        d = euclidean(p, props_raw.centroid)
        d_max = np.max([d_max, d])
        d_min = np.min([d_min, d])

    return d_min, d_max


def choose_objects(img_gray, mask):
    proc = sobel(img_gray, mask=mask > 0)
    proc = scale_image_pixels(proc)

    proc = cv2.equalizeHist(proc)
    _, proc = cv2.threshold(proc, 130, 255, cv2.THRESH_BINARY)

    # open operation
    proc = cv2.dilate(proc, brush_3, iterations=1)
    proc = cv2.erode(proc, brush_3, iterations=1)
    proc = remove_holes(proc)

    proc = cv2.erode(proc, brush_3, iterations=1)
    proc = remove_appendix_objects(proc, 100, 50)

    return proc


def create_features(proc_c, prop_curr):
    _, contours, hierarchy = cv2.findContours(proc_c, 1, 2)
    contour = contours[0]
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True, cv2.CHAIN_APPROX_SIMPLE)

    chosen = filter_hough_lines(proc_c, prop_curr)

    is_convex_1 = is_convex_by_lines_crossing(chosen, prop_curr.coords)
    is_convex_2 = is_convex_by_contour(approx)
    is_convex_3 = is_convex_by_hull(prop_curr)

    r_min, r_max = get_min_max_distance(proc_c)
    r_equiv = prop_curr.equivalent_diameter / 2

    features = {
        'approx_shape': approx.shape[0],
        'num_lines': len(chosen),
        'is_convex': int(np.median([is_convex_1, is_convex_2, is_convex_3])),
        'eccentricity': prop_curr.eccentricity,
        'compactness': prop_curr.perimeter ** 2 / prop_curr.area,
        'solidity': prop_curr.solidity,
        'radius_equiv': r_equiv,
        'radius_min': r_min,
        'radius_max': r_max,
    }

    return features


def extract_features_from_image(img_gray, mask):
    proc = choose_objects(img_gray, mask)

    proc_lbl = label(proc)
    props_reg = {p.label: p for p in regionprops(proc_lbl)}

    df_features = []

    for region_id in range(1, proc_lbl.max() + 1):
        prop_curr = props_reg[region_id]

        proc_c = np.zeros(proc.shape, dtype=np.uint8)
        proc_c[proc_lbl == region_id] = 255

        df_features.append(create_features(proc_c, prop_curr))

    df_features = pd.DataFrame(df_features)
    df_features['line_diff'] = df_features['approx_shape'] - df_features['num_lines']
    df_features['rad_max_div_equiv'] = df_features['radius_max'] / df_features['radius_equiv']
    df_features['rad_min_div_equiv'] = df_features['radius_min'] / df_features['radius_equiv']
    df_features['rad_min_div_max'] = df_features['radius_min'] / df_features['radius_max']

    return df_features, props_reg
