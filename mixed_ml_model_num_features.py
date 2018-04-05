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
import DecisionTree as dt

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
  #data.drop('PPTERM', axis=1, inplace=True)

  # For every feature in the data
  features = data.columns.tolist()
  features.remove("PPTERM")
  ranked_feature_linear = []
  ranked_feature_nmi = []
  for f in features:
    data_f = np.array(data[f])

    # Calculate Linear correlation 
    ranked_feature_linear.append((f, CalcLinearCorrelation(data_f, label)))
    
    # Calculate NMI
    ranked_feature_nmi.append((f, CalcNMI(data_f, label)))

  ranked_feature_linear = sorted(ranked_feature_linear, key=lambda x:x[1], reverse=True)
  ranked_feature_nmi = sorted(ranked_feature_nmi, key=lambda x:x[1], reverse=True)

  feature_list_auc = []
  feature_list_ig = []
  accuracy_dt = []
  auc_dt = []
  fscore_dt = []
  num_features_list = []
  for b in range(1, 11):
    num_feature_auc = 0
    accuracy_auc, auc_auc, fscore_auc, num_feature_auc, feature_list_auc, _ = dt.Run(file_name, "PPTERM", b, "AUC")
    num_feature_ig = 0
    accuracy_ig, auc_ig, fscore_ig, num_feature_ig, feature_list_ig, _ = dt.Run(file_name, "PPTERM", b, "IG")
    accuracy_dt.append([b, accuracy_auc, accuracy_ig])
    auc_dt.append([b, auc_auc, auc_ig])
    fscore_dt.append([b, fscore_auc, fscore_ig])
    num_features_list.append(int(max(num_feature_auc, num_feature_ig)))

  df_accuracy_dt = pd.DataFrame(accuracy_dt, columns=["log(N)", "DT_AUC", "DT_IG"])
  df_auc_dt = pd.DataFrame(auc_dt, columns=["log(N)", "DT_AUC", "DT_IG"])
  df_fscore_dt = pd.DataFrame(fscore_dt, columns=["log(N)", "DT_AUC", "DT_IG"])

  ranked_feature_auc = collections.Counter(feature_list_auc).most_common()
  ranked_feature_ig = collections.Counter(feature_list_ig).most_common()

  print "Number of features: ", num_features_list

  accuracy_lr = []
  auc_lr = []
  fscore_lr = []
  for a in range(1, 11):

    num = num_features_list[a-1]

    features = [i[0] for i in ranked_feature_linear]
    features_selected = features[:num]
    data_for_ml = data[features_selected].as_matrix()
    accuracy_linear, auc_linear, fscore_linear = lr.Run(data_for_ml, label)

    features = [i[0] for i in ranked_feature_nmi]
    features_selected = features[:num]
    data_for_ml = data[features_selected].as_matrix()
    accuracy_nmi, auc_nmi, fscore_nmi = lr.Run(data_for_ml, label)

    accuracy_lr.append([a, accuracy_linear, accuracy_nmi])
    auc_lr.append([a, auc_linear, auc_nmi])
    fscore_lr.append([a, fscore_linear, fscore_nmi])

  df_accuracy_lr = pd.DataFrame(accuracy_lr, columns=["log(N)", "Linear", "NMI"])
  df_auc_lr = pd.DataFrame(auc_lr, columns=["log(N)", "Linear", "NMI"])
  df_fscore_lr = pd.DataFrame(fscore_lr, columns=["log(N)", "Linear", "NMI"])

  df_accuracy = df_accuracy_dt.merge(df_accuracy_lr, on="log(N)")
  df_auc = df_auc_dt.merge(df_auc_lr, on="log(N)")
  df_fscore = df_fscore_dt.merge(df_fscore_lr, on="log(N)")
  print df_accuracy
  print df_auc
  print df_fscore
  df_accuracy.to_csv("Accuracy"+"_"+time.strftime("%m%d%Y")+".csv", index=False)
  df_auc.to_csv("AUC"+"_"+time.strftime("%m%d%Y")+".csv", index=False)
  df_fscore.to_csv("Fscore"+"_"+time.strftime("%m%d%Y")+".csv", index=False)

  num_top_features = 50
  top_feature_linear = [i[0] for i in ranked_feature_linear[:num_top_features]]
  top_feature_nmi = [i[0] for i in ranked_feature_nmi[:num_top_features]]
  top_feature_auc = [i[0] for i in ranked_feature_auc[:num_top_features]]
  top_feature_ig = [i[0] for i in ranked_feature_ig[:num_top_features]]
  df_selected_features = pd.DataFrame({"Linear":top_feature_linear, "NMI":top_feature_nmi, "AUC":top_feature_auc, "IG":top_feature_ig})
  print df_selected_features
  df_selected_features.to_csv("SelectedFeatures"+"_"+time.strftime("%m%d%Y")+".csv", index=False)

  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
