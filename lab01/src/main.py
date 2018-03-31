#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import joblib

import cv2

from multiprocessing import Pool

from masks.cards import create_mask_cards
from masks.figures import create_mask_figures
from features import extract_features_from_image


def format_answer(is_poly, is_convex, num_vertices):
    marker = []
    if is_poly:
        marker.append('P')
        marker.append(str(num_vertices))
        if is_convex:
            marker.append('C')
    else:
        marker.append('S')
    return ''.join(marker)


def process_image(image_src, image_dst):
    global clf

    img_gray = cv2.imread(image_src, 0)
    img_color = cv2.imread(image_src, 1)

    mask_cards = create_mask_cards(img_color)
    mask_figures = create_mask_figures(img_gray, mask_cards)
    features, props = extract_features_from_image(img_gray, mask_figures)

    y_pred = clf.predict(features.values)

    for obj_i, region_id in enumerate(range(1, features.shape[0] + 1)):
        prop_curr = props[region_id]

        y_ = y_pred[obj_i]
        is_convex = features.loc[obj_i, 'is_convex']
        is_poly = y_ != 0

        marker = format_answer(is_poly, is_convex, y_)

        y_cent, x_cent = map(int, prop_curr.centroid)
        cv2.rectangle(
            img_color,
            (x_cent - len(marker) * 10 - 5, y_cent + 18),
            (x_cent + len(marker) * 10 + 5, y_cent - 20),
            (255, 255, 255), -1
        )
        cv2.putText(
            img_color, marker, (x_cent - len(marker) * 10, y_cent + 10),
            1, 2, (0, 0, 255), 2, cv2.LINE_AA
        )

    cv2.imwrite(image_dst, img_color)


def process_image_wrapper(args):
    process_image(*args)


if __name__ == "__main__":
    src_dir, dst_dir = sys.argv[1:]

    images = filter(lambda x: os.path.isfile(os.path.join(src_dir, x)) and \
                              os.path.splitext(x)[1], os.listdir(src_dir))

    images_src = map(lambda x: os.path.join(src_dir, x), images)
    images_dst = map(lambda x: os.path.join(dst_dir, x), images)

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    clf = joblib.load(os.path.join(src_dir, '../data/clf_model_vertices.model'))

    pool = Pool(processes=min(len(images), 5))
    pool.map(process_image_wrapper, zip(images_src, images_dst))
    pool.close()
