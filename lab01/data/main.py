#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df_features = [
    pd.read_csv('ml_set_features/features_{:02}.csv'.format(i), sep=',', encoding='utf-8')
    for i in [1, 2, 4, 6, 7, 9, 10]
]

df_features = pd.concat(df_features)
columns = filter(lambda x: x != 'img', df_features.columns)
df_features = df_features.loc[:, ['img'] + columns]
df_features = df_features.loc[df_features.loc[:, 'img'] != '09_18']  # removed spoiled image

df_features.to_csv("ml_set_features.csv", sep=',', encoding='utf-8', index=False)

df_labels = pd.read_csv('ml_set_labels.csv')

X_train, X_valid, Y_train, Y_valid = \
    train_test_split(df_features.values[:, 1:], df_labels['VERT'], test_size=0.1)

clf = RandomForestClassifier(n_estimators=100, verbose=0, criterion='entropy')
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_valid)

print accuracy_score(Y_valid, Y_pred)

joblib.dump(clf, 'clf_model_vertices.model')
