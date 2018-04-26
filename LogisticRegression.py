#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import collections
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics

import NMI as nmi
import LinearCorr as linear


def Run(data, num_features, label_name, ranking_method):
  features = data.columns.tolist()
  indices = data.index.tolist()
  features.remove(label_name)
  x = data[features]
  y = data[label_name].as_matrix()

  n_folds = 10
  accuracy_list = []
  false_positive_rate_list = []
  false_negative_rate_list = []
  true_positive_rate_list = []
  auc_list = []
  fscore_list = []
  cv_arg = KFold(n_folds, shuffle=True)
  num_rounds = 0
  for train_idx, test_idx in cv_arg.split(x.as_matrix()):
    # x is data frame while y is numpy array
    ranked_features = []
    selected_features = []
    train_set = x.iloc[train_idx]
    train_label = y[train_idx]
    test_set = x.iloc[test_idx]
    ref = y[test_idx]

    for f in features:
      data_f = train_set[f].as_matrix()
      if ranking_method == "NMI":
        ranked_features.append((f, nmi.Calc(data_f, train_label)))
      elif ranking_method == "Linear":
        ranked_features.append((f, linear.Calc(data_f, train_label)))
      else:
        print "Unrecognized feature ranking method"
        exit()
    ranked_features = sorted(ranked_features, key=lambda x:x[1], reverse=True)
    selected_features = [f_i[0] for f_i in ranked_features]
    selected_features = selected_features[:num_features]
    train_set_array = train_set[selected_features].as_matrix()
    test_set_array = test_set[selected_features].as_matrix()

    logreg = linear_model.LogisticRegression()
    logreg.fit(train_set_array, train_label)
    results = logreg.predict(test_set_array)
    count = 0
    true_positive_count = 0
    false_negative_count = 0
    false_positive_count = 0
    preterm_count = 0
    term_count = 0
    for i in range(len(results)):
      if results[i] == ref[i]:
        count += 1
      if ref[i] == 2:
        preterm_count += 1
      if ref[i] == 1:
        term_count += 1
      # False negative
      if ref[i] == 2 and results[i] == 1:
        false_negative_count += 1
      # False positive
      if ref[i] == 1 and results[i] == 2:
        false_positive_count += 1
      # True positive
      if ref[i] == 2 and results[i] == 2:
        true_positive_count += 1
    if preterm_count == 0:
      continue
    num_rounds += 1
    accuracy = float(count) / float(len(results))
    false_negative_rate = float(false_negative_count) / float(preterm_count)
    false_positive_rate = float(false_positive_count) / float(term_count)
    true_positive_rate = float(true_positive_count) / float(preterm_count)

    accuracy_list.append(accuracy)
    false_negative_rate_list.append(false_negative_rate)
    false_positive_rate_list.append(false_positive_rate)
    true_positive_rate_list.append(true_positive_rate)
    auc_list.append(metrics.auc([0.0, false_positive_rate, 1.0], [0.0, true_positive_rate, 1.0]))
    fscore_list.append(metrics.fbeta_score(ref, results, 2))
        
  return (accuracy_list, auc_list, fscore_list)


