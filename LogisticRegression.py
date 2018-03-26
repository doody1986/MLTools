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


def Run(data, label):
  x = data
  y = label
  num_features = x.shape[1]
  num_samples = x.shape[0]

  n_folds = 10
  final_accuracy = 0.0
  final_false_positive_rate = 0.0
  final_false_negative_rate = 0.0
  final_true_positive_rate = 0.0
  cv_arg = KFold(n_folds, shuffle=True)
  num_rounds = 0
  for train_idx, test_idx in cv_arg.split(x):
    train_set = x[train_idx]
    train_label = y[train_idx]
    test_set = x[test_idx]
    ref = y[test_idx]
    
    logreg = linear_model.LogisticRegression()
    logreg.fit(train_set, train_label)
    result = logreg.predict(test_set)
    count = 0
    true_positive_count = 0
    false_negative_count = 0
    false_positive_count = 0
    preterm_count = 0
    term_count = 0
    for i in range(len(result)):
      if result[i] == ref[i]:
        count += 1
      if ref[i] == 2:
        preterm_count += 1
      if ref[i] == 1:
        term_count += 1
      # False negative
      if ref[i] == 2 and result[i] == 1:
        false_negative_count += 1
      # False positive
      if ref[i] == 1 and result[i] == 2:
        false_positive_count += 1
      # True positive
      if ref[i] == 2 and result[i] == 2:
        true_positive_count += 1
    if preterm_count == 0:
      continue
    num_rounds += 1
    final_accuracy += float(count) / float(len(result))
    final_false_negative_rate += float(false_negative_count) / float(preterm_count)
    final_false_positive_rate += float(false_positive_count) / float(term_count)
    final_true_positive_rate += float(true_positive_count) / float(preterm_count)

  final_accuracy /= num_rounds
  final_false_negative_rate /= num_rounds
  final_false_positive_rate /= num_rounds
  final_true_positive_rate /= num_rounds
  final_auc = metrics.auc([0.0, final_false_positive_rate, 1.0], [0.0, final_true_positive_rate, 1.0])
        
  #print "Averaged metrics"
  #print final_accuracy
  #print final_false_negative_rate
  #print final_false_positive_rate
  #print "Final AUC: ", metrics.auc([0.0, final_false_positive_rate, 1.0], [0.0, final_true_positive_rate, 1.0])

  return (final_accuracy, final_auc)


