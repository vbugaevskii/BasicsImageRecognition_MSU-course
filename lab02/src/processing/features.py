#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import *

from skimage import transform as tf

from skimage.measure import label, regionprops
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel

from pywt import dwt2


def vertices2chain(vertices):
    vertices = np.array(vertices, dtype=np.float32, ndmin=2)
    assert vertices.shape[1] == 2
    chain = vertices[1:,] - vertices[:-1,]
    return chain


def chain_parts_length(vertices):
    chain = vertices2chain(vertices)
    chain_length = np.linalg.norm(chain, axis=1)
    return chain_length


def measure_angle_between(v1, v2, oriented=False):
    def norm(vector):
        return vector / np.linalg.norm(vector)

    v1, v2 = norm(v1), norm(v2)
    vcos = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(vcos)

    if oriented:
        vsin = np.clip(np.cross(v1, v2), -1.0, 1.0)
        if vsin < 0:
            angle = -angle

    return angle


def chain_parts_angles(vertices, oriented=False):
    chain = vertices2chain(vertices)
    angles = [measure_angle_between(chain[i], chain[i + 1], oriented)
              for i in range(len(chain) - 1)]
    return angles


def fingers_features(img_fingers):
    assert check_image_consistency(img_fingers)

    """
    Двойное транспонирование для того, чтобы пальцы шли в следующем порядке:
        мезинец        1
        безымянный     2
        средний        3
        указательный   4
        большой        5
    """
    img_labels = label(img_fingers.T).T
    img_props = regionprops(img_labels)

    features = []
    for prop in img_props:
        prop_f = (prop.major_axis_length, prop.minor_axis_length)
        features.append(prop_f)
    return features


def incircle(img):
    assert check_image_consistency(img)

    distance = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    row_center, col_center = np.unravel_index(
        np.argmax(distance, axis=None), distance.shape)
    radius = distance[row_center, col_center]

    return (col_center, row_center), radius


def stats_mask_pixels_intesity(img_gray, img_mask, func):
    assert check_image_consistency(img_mask)
    return func(img_gray[np.where(img_mask > 0)])


def mean_hand_pixels_intensity(img_gray, img_hand):
    return stats_mask_pixels_intesity(img_gray, img_hand, np.mean)


def median_hand_pixels_intensity(img_gray, img_hand):
    return stats_mask_pixels_intesity(img_gray, img_hand, np.median)


def percentile_hand_pixels_intensity(img_gray, img_hand, percentile):
    return stats_mask_pixels_intesity(img_gray, img_hand, lambda x: np.percentile(x, percentile))


def wavelet_transform(img, wavelet='haar', steps=2, pool=4):
    tempf = np.zeros_like(img)
    energies = []

    def filter_frequencies(img, pcnt_low, pcnt_high):
        assert pcnt_low < pcnt_high

        img = img.copy()
        low, high = np.percentile(img, [pcnt_low, pcnt_high])
        img[np.where(np.logical_and(low < img, img < high))] = 0
        return img

    LL = img / 255.0
    for i in range(steps):
        LL, (LH, HL, HH) = dwt2(LL, wavelet=wavelet)

#         LH = filter_frequencies(LH, 5, 95)
#         HL = filter_frequencies(HL, 5, 95)
#         HH = filter_frequencies(HH, 5, 95)

        row_step, col_step = LL.shape[0] // pool, LL.shape[1] // pool
        for row in range(0, LL.shape[0], row_step):
            for col in range(0, LL.shape[1], col_step):
                energies.extend(map(lambda x: np.power(x, 2.0).sum(), [
                    LH[row:row + row_step, col:col + col_step],
                    HL[row:row + row_step, col:col + col_step],
                    HH[row:row + row_step, col:col + col_step]
                ]))

        LH, HL, HH = map(scale_image_pixels, map(np.abs, [LH, HL, HH]))

        tempf[0:LL.shape[0], 0:LL.shape[1]] = LL
        tempf[0:LH.shape[0], LL.shape[1]:LL.shape[1] + LH.shape[1]] = LH
        tempf[LL.shape[0]:LL.shape[0] + HL.shape[0], 0:HL.shape[1]] = HL
        tempf[LL.shape[0]:LL.shape[0] + HH.shape[0], LL.shape[1]:LL.shape[1] + HH.shape[1]] = HH

    tempf[0:LL.shape[0], 0:LL.shape[1]] = scale_image_pixels(LL)
    tempf = scale_image_pixels(tempf)

    return tempf, energies


def texture_features(img_gray, img_palm, tform):
    img_props = regionprops(label(img_palm))[0]
    row_center, col_center = map(int, img_props.centroid)
    row_center -= 10
    col_center -= 10

    img_patch = tf.warp(img_gray, tform)
    img_patch = scale_image_pixels(img_patch)
    img_patch = img_patch[row_center - 64:row_center + 64, col_center - 64:col_center + 64]

    img_patch, energies = wavelet_transform(img_patch)

    features = dict()

    angles = [0, 90]
    glcm = greycomatrix(img_patch, [5], angles, 256, symmetric=True, normed=True)
    for prop in ['contrast', 'energy', 'dissimilarity', 'correlation']:
        for alpha, val in zip(angles, greycoprops(glcm, prop)[0]):
            features['texture_{}_{:02d}'.format(prop, alpha)] = val

    # features.update({"wavelet_{}".format(i): e for i, e in enumerate(energies)})

    return features, img_patch


def extract_features(img_gray, tform, img_hand, img_palm, img_fingers, vertices):
    features = dict()

    chains = chain_parts_length(vertices)
    for chain_i, chain_len in enumerate(chains):
        features['chain_{}_len'.format(chain_i)] = chain_len

    fingers = fingers_features(img_fingers)
    for finger_i, (major_len, minor_len) in enumerate(fingers):
        features['fingure_{}_len'.format(finger_i)] = major_len
        features['fingure_{}_width'.format(finger_i)] = minor_len

    _, radius = incircle(img_palm)
    features['palm_circle_radius'] = radius

    img_mask = tf.warp(img_hand, tform.inverse)
    img_mask = scale_image_pixels(img_mask)
    features['median_skin_color'] = median_hand_pixels_intensity(img_gray, img_mask)

    """
    for pcnt in [10, 50, 90]:
        features['pcnt_{}_skin_color'.format(pcnt)] = \
            percentile_hand_pixels_intensity(img_gray, img_mask, percentile=pcnt)
    """

    features_tex, _ = texture_features(img_gray, img_palm, tform)
    features.update(features_tex)

    return features


def palm_roi(img_color, img_palm, tform):
    img_props = regionprops(label(img_palm))[0]
    row_min, col_min, row_max, col_max = img_props.bbox

    img_patch = tf.warp(img_color, tform)
    img_patch = img_patch[row_min:row_max, col_min:col_max]
    img_mask  = img_palm[row_min:row_max, col_min:col_max]

    img_patch = cv2.bitwise_and(img_patch, img_patch, mask=img_mask)
    img_patch *= 255
    img_patch = img_patch.astype(np.uint8)

    img_patch = padding(img_patch, (490, 490))
    return img_patch


def palm_roi_64(img_color, img_palm, tform):
    img_props = regionprops(label(img_palm))[0]
    row_center, col_center = map(int, img_props.centroid)
    row_center -= 10
    col_center -= 10

    img_patch = tf.warp(img_color, tform)
    img_patch *= 255
    img_patch = img_patch[row_center - 64:row_center + 64, col_center - 64:col_center + 64]
    return img_patch
