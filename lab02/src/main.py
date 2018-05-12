#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import cv2

import pandas as pd

from multiprocessing import Pool

from processing import extract_tips_and_valleys, draw_vertices_on_image
from processing.masks import create_hand_mask, create_palm_and_fingers_masks, position_hand_mask

from processing.utils.transform import transform_vertices

from processing.features import extract_features, palm_roi, palm_roi_64


def process_image(img_src, img_dst, img_ftr):
    img_color = cv2.imread(img_src, 1)
    img_gray  = cv2.imread(img_src, 0)

    img_hand = create_hand_mask(img_gray)
    img_hand, tform = position_hand_mask(img_hand)
    img_hand, img_palm, img_fingers = create_palm_and_fingers_masks(img_hand)
    vertices = extract_tips_and_valleys(img_hand, img_palm, img_fingers)
    vertices = transform_vertices(vertices, tform)

    img_color_res = draw_vertices_on_image(img_color, vertices)
    cv2.imwrite(img_dst, img_color_res)

    # img_color_res = palm_roi(img_color, img_hand, tform)
    # img_color_res = palm_roi_64(img_color, img_palm, tform)
    # cv2.imwrite(img_ftr, img_color_res)

    img_features = extract_features(img_gray, tform, img_hand, img_palm, img_fingers, vertices)
    img_features['img'] = img_src
    return img_features


def process_image_wrapper(args):
    img_src = args[0]
    res = None
    try:
        res = process_image(*args)
        print "'{}' is successfully processed".format(img_src)
    except AssertionError as e:
        print "failed to process '{}'!".format(img_src)
    return res


if __name__ == "__main__":
    """
    src_dir  -- директория с исходными изображениями;
    dst_dir  -- директория, куда сохраняются размеченные изображения;
    data_dir -- директория, куда сохраняются признаки для кластеризации.
    """
    src_dir, dst_dir, data_dir = sys.argv[1:]

    images = filter(lambda x: os.path.isfile(os.path.join(src_dir, x)) and \
                              os.path.splitext(x)[1], os.listdir(src_dir))

    img_features_dir = os.path.join(data_dir, 'img')
    images_src = map(lambda x: os.path.join(src_dir, x), images)
    images_dst = map(lambda x: os.path.join(dst_dir, x), images)
    images_ftr = map(lambda x: os.path.join(img_features_dir, x), images)

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if not os.path.isdir(img_features_dir):
        os.mkdir(img_features_dir)

    pool = Pool(processes=min(len(images), 5))
    df_features = pool.map(process_image_wrapper, zip(images_src, images_dst, images_ftr))
    df_features = pd.DataFrame(df_features)
    df_features.sort_values(by="img", inplace=True)
    df_features.to_csv(os.path.join(data_dir, "ml_set_features.csv"), index=False, encoding='utf-8')
    pool.close()
