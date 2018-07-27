#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import time
import re
import os
import collections

from sklearn import linear_model
from sklearn import metrics

import LogisticRegression as lr
import DecisionTree as dt
import NMI as nmi
import LinearCorr as linear

current_path = os.getcwd()

def main():
  print ("Start program.")

  if len(sys.argv) < 2:
    print "Too few arguments"
    print "Please specify the data csv files."
    sys.exit()
  file_name = sys.argv[1]

  data = pd.read_csv(file_name)

  max_num_features = 100
  num_features = range(10, max_num_features, 10)

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
    ranked_feature_linear.append((f, linear.Calc(data_f, label)))
    
    # Calculate NMI
    ranked_feature_nmi.append((f, nmi.Calc(data_f, label)))

  ranked_feature_linear = sorted(ranked_feature_linear, key=lambda x:x[1], reverse=True)
  ranked_feature_nmi = sorted(ranked_feature_nmi, key=lambda x:x[1], reverse=True)

  ranked_feature_auc = []
  ranked_feature_ig = []
  ranked_feature_auc = dt.SelectFeature(data, "PPTERM", "AUC", max_num_features)
  ranked_feature_ig = dt.SelectFeature(data, "PPTERM", "IG", max_num_features)

  #accuracy_dt = []
  #auc_dt = []
  #fscore_dt = []
  #accuracy_var_dt = []
  #auc_var_dt = []
  #fscore_var_dt = []
  #accuracy_auc, _, _, auc_auc, fscore_auc = dt.Run(data, "PPTERM", max_num_features, "AUC")
  #accuracy_ig, _, _, auc_ig, fscore_ig = dt.Run(data, "PPTERM", max_num_features, "IG")

  #for b in num_features:
  #  # Calculate the mean and variance
  #  accuracy_auc_mean = np.mean(np.array(accuracy_auc)[:, b])
  #  accuracy_auc_var = np.var(np.array(accuracy_auc)[:, b])
  #  auc_auc_mean = np.mean(np.array(auc_auc)[:, b])
  #  auc_auc_var = np.var(np.array(auc_auc)[:, b])
  #  fscore_auc_mean = np.mean(np.array(fscore_auc)[:, b])
  #  fscore_auc_var = np.var(np.array(fscore_auc)[:, b])
  #  accuracy_ig_mean = np.mean(np.array(accuracy_ig)[:, b])
  #  accuracy_ig_var = np.var(np.array(accuracy_ig)[:, b])
  #  auc_ig_mean = np.mean(np.array(auc_ig)[:, b])
  #  auc_ig_var = np.var(np.array(auc_ig)[:, b])
  #  fscore_ig_mean = np.mean(np.array(fscore_ig)[:, b])
  #  fscore_ig_var = np.var(np.array(fscore_ig)[:, b])

  #  # create the list for generating the table
  #  accuracy_dt.append([b+1, accuracy_auc_mean, accuracy_ig_mean])
  #  auc_dt.append([b+1, auc_auc_mean, auc_ig_mean])
  #  fscore_dt.append([b+1, fscore_auc_mean, fscore_ig_mean])
  #  accuracy_var_dt.append([b+1, accuracy_auc_var, accuracy_ig_var])
  #  auc_var_dt.append([b+1, auc_auc_var, auc_ig_var])
  #  fscore_var_dt.append([b+1, fscore_auc_var, fscore_ig_var])

  #df_accuracy_dt = pd.DataFrame(accuracy_dt, columns=["N", "DT_AUC", "DT_IG"])
  #df_auc_dt = pd.DataFrame(auc_dt, columns=["N", "DT_AUC", "DT_IG"])
  #df_fscore_dt = pd.DataFrame(fscore_dt, columns=["N", "DT_AUC", "DT_IG"])
  #df_accuracy_var_dt = pd.DataFrame(accuracy_var_dt, columns=["N", "DT_AUC", "DT_IG"])
  #df_auc_var_dt = pd.DataFrame(auc_var_dt, columns=["N", "DT_AUC", "DT_IG"])
  #df_fscore_var_dt = pd.DataFrame(fscore_var_dt, columns=["N", "DT_AUC", "DT_IG"])

  #accuracy_lr = []
  #auc_lr = []
  #fscore_lr = []
  #accuracy_var_lr = []
  #auc_var_lr = []
  #fscore_var_lr = []
  #for a in num_features:

  #  accuracy_linear, auc_linear, fscore_linear = lr.Run(data, a+1, "PPTERM", "Linear")

  #  # Calculate the mean and variance for linear correlation
  #  accuracy_linear_mean = np.mean(np.array(accuracy_linear))
  #  accuracy_linear_var = np.var(np.array(accuracy_linear))
  #  auc_linear_mean = np.mean(np.array(auc_linear))
  #  auc_linear_var = np.var(np.array(auc_linear))
  #  fscore_linear_mean = np.mean(np.array(fscore_linear))
  #  fscore_linear_var = np.var(np.array(fscore_linear))

  #  accuracy_nmi, auc_nmi, fscore_nmi = lr.Run(data, a+1, "PPTERM", "NMI")

  #  # Calculate the mean and variance for nmi
  #  accuracy_nmi_mean = np.mean(np.array(accuracy_nmi))
  #  accuracy_nmi_var = np.var(np.array(accuracy_nmi))
  #  auc_nmi_mean = np.mean(np.array(auc_nmi))
  #  auc_nmi_var = np.var(np.array(auc_nmi))
  #  fscore_nmi_mean = np.mean(np.array(fscore_nmi))
  #  fscore_nmi_var = np.var(np.array(fscore_nmi))

  #  accuracy_lr.append([a+1, accuracy_linear_mean, accuracy_nmi_mean])
  #  auc_lr.append([a+1, auc_linear_mean, auc_nmi_mean])
  #  fscore_lr.append([a+1, fscore_linear_mean, fscore_nmi_mean])
  #  accuracy_var_lr.append([a+1, accuracy_linear_var, accuracy_nmi_var])
  #  auc_var_lr.append([a+1, auc_linear_var, auc_nmi_var])
  #  fscore_var_lr.append([a+1, fscore_linear_var, fscore_nmi_var])

  #df_accuracy_lr = pd.DataFrame(accuracy_lr, columns=["N", "Linear", "NMI"])
  #df_auc_lr = pd.DataFrame(auc_lr, columns=["N", "Linear", "NMI"])
  #df_fscore_lr = pd.DataFrame(fscore_lr, columns=["N", "Linear", "NMI"])
  #df_accuracy_var_lr = pd.DataFrame(accuracy_var_lr, columns=["N", "Linear", "NMI"])
  #df_auc_var_lr = pd.DataFrame(auc_var_lr, columns=["N", "Linear", "NMI"])
  #df_fscore_var_lr = pd.DataFrame(fscore_var_lr, columns=["N", "Linear", "NMI"])

  #df_accuracy = df_accuracy_dt.merge(df_accuracy_lr, on="N")
  #df_auc = df_auc_dt.merge(df_auc_lr, on="N")
  #df_fscore = df_fscore_dt.merge(df_fscore_lr, on="N")
  #df_accuracy_var = df_accuracy_var_dt.merge(df_accuracy_var_lr, on="N")
  #df_auc_var = df_auc_var_dt.merge(df_auc_var_lr, on="N")
  #df_fscore_var = df_fscore_var_dt.merge(df_fscore_var_lr, on="N")
  #print df_accuracy
  #print df_auc
  #print df_fscore
  #df_accuracy.to_csv(current_path+"/Results/Accuracy"+"_"+time.strftime("%m%d%Y")+".csv", index=False)
  #df_auc.to_csv(current_path+"/Results/AUC"+"_"+time.strftime("%m%d%Y")+".csv", index=False)
  #df_fscore.to_csv(current_path+"/Results/Fscore"+"_"+time.strftime("%m%d%Y")+".csv", index=False)
  #df_accuracy_var.to_csv(current_path+"/Results/Accuracy_var"+"_"+time.strftime("%m%d%Y")+".csv", index=False)
  #df_auc_var.to_csv(current_path+"/Results/AUC_var"+"_"+time.strftime("%m%d%Y")+".csv", index=False)
  #df_fscore_var.to_csv(current_path+"/Results/Fscore_var"+"_"+time.strftime("%m%d%Y")+".csv", index=False)

  num_selected_features = 20
  top_feature_linear = [i[0] for i in ranked_feature_linear[:num_selected_features]]
  top_feature_nmi = [i[0] for i in ranked_feature_nmi[:num_selected_features]]
  top_feature_auc = [i[0] for i in ranked_feature_auc[:num_selected_features]]
  top_feature_ig = [i[0] for i in ranked_feature_ig[:num_selected_features]]
  print top_feature_linear
  print top_feature_nmi
  print top_feature_auc
  print top_feature_ig
  df_selected_features = pd.DataFrame({"Linear":top_feature_linear, "NMI":top_feature_nmi, "AUC":top_feature_auc, "IG":top_feature_ig})
  print df_selected_features
  df_selected_features.to_csv(current_path+"/Results/SelectedFeatures"+"_"+time.strftime("%m%d%Y")+".csv", index=False)

  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
