#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import time
import re
import collections

from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn import metrics

import LogisticRegression as lr

def CalcLinearCorrelation(data, label):
  null_flags = np.isnan(data).tolist()
  valid_idx = [idx for idx, value in enumerate(null_flags) if value == False]
  local_data = data[[valid_idx]]
  local_label = label[[valid_idx]]
  temp = pearsonr(local_data, local_label)
  result = abs(temp[0])
  if np.isnan(result):
    result = 0
  return result

def CalcNMI(data, label):
  null_flags = np.isnan(data).tolist()
  valid_idx = [idx for idx, value in enumerate(null_flags) if value == False]
  local_data = data[[valid_idx]]
  local_label = label[[valid_idx]]
  result = normalized_mutual_info_score(local_data, local_label)
  return result

def main():
  print ("Start program.")

  if len(sys.argv) < 2:
    print "Too few arguments"
    print "Please specify the data csv files."
    sys.exit()
  file_name = sys.argv[1]

  data = pd.read_csv(file_name)

  # Get Label
  label = np.array(data['PPTERM'])
  data.drop('PPTERM', axis=1, inplace=True)

  # For every feature in the data
  features = data.columns.tolist()
  ranked_feature_linear = []
  ranked_feature_nmi = []
  for f in features:
    data_f = np.array(data[f])

    # Calculate Linear correlation 
    ranked_feature_linear.append((f, CalcLinearCorrelation(data_f, label)))
    ranked_feature_linear = sorted(ranked_feature_linear, key=lambda x:x[1])
    
    # Calculate NMI
    ranked_feature_nmi.append((f, CalcNMI(data_f, label)))
    ranked_feature_nmi = sorted(ranked_feature_nmi, key=lambda x:x[1])

  accuracy_linear = []
  accuracy_nmi = []
  auc_linear = []
  auc_nmi = []
  for num in range(5, 300, 5):
    features = [i[0] for i in ranked_feature_linear]
    features_selected = features[:num]
    data_for_ml = data[features_selected].as_matrix()
    accuracy, auc = lr.Run(data_for_ml, label)
    accuracy_linear.append((num, accuracy))
    auc_linear.append((num, auc))

    features = [i[0] for i in ranked_feature_nmi]
    features_selected = features[:num]
    data_for_ml = data[features_selected].as_matrix()
    accuracy, auc = lr.Run(data_for_ml, label)
    accuracy_nmi.append((num, accuracy))
    auc_nmi.append((num, auc))

  print accuracy_linear
  print accuracy_nmi
  print auc_linear
  print auc_nmi

  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
