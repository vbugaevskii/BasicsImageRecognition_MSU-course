#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


def find_intersection_between_lines(rho_1, theta_1, rho_2, theta_2):
    if np.isclose(theta_1, theta_2):
        return None if not np.isclose(rho_1, rho_2) else np.array([np.nan, np.nan])

    alpha_1 = np.array([np.cos(theta_1), np.sin(theta_1)]) * rho_1
    beta_1 = np.array([np.sin(theta_1), -np.cos(theta_1)])

    alpha_2 = np.array([np.cos(theta_2), np.sin(theta_2)]) * rho_2
    beta_2 = np.array([np.sin(theta_2), -np.cos(theta_2)])

    A = np.hstack([beta_1.reshape(-1, 1), -beta_2.reshape(-1, 1)])
    b = alpha_2 - alpha_1
    t = np.linalg.solve(A, b)

    return alpha_1 + beta_1 * t[0]


def find_angle_between_lines(rho_1, theta_1, rho_2, theta_2):
    alpha_1 = np.array([np.cos(theta_1), np.sin(theta_1)]) * rho_1
    alpha_2 = np.array([np.cos(theta_2), np.sin(theta_2)]) * rho_2

    v1_u = alpha_1 / np.linalg.norm(alpha_1)
    v2_u = alpha_2 / np.linalg.norm(alpha_2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle if angle < np.pi / 2 else np.pi - angle


def filter_hough_lines(proc_c, prop_raw):
    edges = canny(proc_c, sigma=0.1)

    hspace, angles, dists = hough_line(edges)
    hspace, angles, dists = hough_line_peaks(hspace, angles, dists, min_angle=1, min_distance=10)

    return zip(dists, angles)

# Deprecated function
"""
def filter_hough_lines(proc_c, prop_raw):
    edges = canny(proc_c, sigma=0.1)

    hspace, angles, dists = hough_line(edges)
    hspace, angles, dists = hough_line_peaks(hspace, angles, dists, min_angle=1, min_distance=10)

    chosen = []

    for rho, theta in zip(dists, angles):
        is_good = True

        for rho_, theta_ in chosen:
            d = find_intersection_between_lines(rho_, theta_, rho, theta)
            a = find_angle_between_lines(rho_, theta_, rho, theta)

            if d is None:
                is_good = not np.abs(rho_ - rho) < 20
                break
            elif np.isnan(d).any():
                is_good = False
                break

            d = np.sqrt(np.linalg.norm(d - prop_raw.centroid))
            if d < 38 and a < np.pi / 180 * 10:
                is_good = False
                break

        if is_good:
            chosen.append((rho, theta))
        else:
            continue
    # print '{} lines chosen from {} variants'.format(len(chosen), len(dists))
    return chosen
"""
