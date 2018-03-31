#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import sys

import cv2

from multiprocessing import Pool

from masks.cards import create_mask_cards
from masks.figures import create_mask_figures
from features import extract_features_from_image


def process_image(image_src):
    img_gray = cv2.imread(image_src, 0)
    img_color = cv2.imread(image_src, 1)

    mask_cards = create_mask_cards(img_color)
    mask_figures = create_mask_figures(img_gray, mask_cards)
    features, _ = extract_features_from_image(img_gray, mask_figures)

    img_src_index = int(re.search(r'\d+', os.path.basename(image_src)).group(0))
    features['img'] = map(lambda x: '{:02}_{:02}'.format(img_src_index, x+1), features.index)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_name = '../data/ml_set_features/features_{:0>2}.csv'.format(img_src_index)
    features.to_csv(os.path.join(script_dir, output_name), sep=',', index=False)


if __name__ == '__main__':
    src_dir = sys.argv[1]

    images = filter(lambda x: os.path.isfile(os.path.join(src_dir, x)) and \
                              os.path.splitext(x)[1], os.listdir(src_dir))
    images_src = map(lambda x: os.path.join(src_dir, x), images)

    pool = Pool(processes=min(len(images), 5))
    pool.map(process_image, images_src)
    pool.close()
