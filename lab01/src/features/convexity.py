#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def get_side_from_line(rho, theta, point):
    n = np.array([rho * np.cos(theta), rho * np.sin(theta)])
    # ATTENTION: point = (row, column) = (y, x)
    return np.dot(n, point[::-1] - n)


def is_not_crossing_line(rho, theta, points):
    score = np.array([get_side_from_line(rho, theta, p) for p in points])
    score_p, score_n = (score > 0).sum(), (score < 0).sum()
    score = min(score_p, score_n) / float(score_p + score_n)
    return score < 0.05


def is_convex_by_lines_crossing(lines, points):
    score = [is_not_crossing_line(rho, theta, points) for rho, theta in lines]
    return all(score)


def is_convex_by_contour(contour):
    contour_ = contour.reshape(-1, 2).tolist()
    contour_.append(contour_[0])
    contour_ = np.array(contour_)

    v = (contour_[1:] - contour_[:-1]).tolist()
    v = [np.cross(v1, v2) > 0 for v1, v2 in zip(v[:-1], v[1:])]
    return sum(v) == len(v) or sum(v) == 0


def is_convex_by_hull(prop):
    return prop.solidity > 0.93

