#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

f_features = sys.argv[1]

df = pd.read_csv(f_features)
df["img"] = df["img"].apply(lambda s: os.path.splitext(os.path.basename(s))[0])
features = filter(lambda x: x != "img", df.columns)
df = df[["img"] + features]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features].values)

distances = cdist(df[features], df[features])
index = np.argsort(distances, axis=1)[:, 1:4]

print "[CLOSEST PALMS]"
for i, img_name in enumerate(df["img"]):
    commons = u' '.join(df["img"].iloc[index[i]].tolist())
    print img_name + u'\t' + commons
print "=" * 80

n_clusters = 20

cl = KMeans(n_clusters=n_clusters, max_iter=2000, tol=1e-6, random_state=8001)
cl.fit(df[features].values)
Y_pred = cl.predict(df[features].values)
df_users = pd.DataFrame(zip(df["img"], Y_pred), columns=["img", "label"])
df_users = df_users.groupby("label")

print "[CLUSTER PALMS]"
for y, group in df_users:
    commons = u' '.join(sorted(group["img"].unique()))
    print u"person_{:02d}".format(y + 1) + u'\t' + commons
