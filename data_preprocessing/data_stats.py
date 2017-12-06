#! /usr/bin/env python
import sys 
import numpy as np
import pandas as pd
import collections
import csv

def main():
  num_args = len(sys.argv)
  if num_args < 2:
    print "There should be at least one input data file!"
    exit()
  arg_list = sys.argv[1:]
  feature_list = []
  study_id_list = []
  df_dict = collections.OrderedDict()
  for file_name in arg_list:
    df_dict[file_name] = pd.read_csv(file_name)
    feature_list += list(df_dict[file_name].columns.values)
    study_id_list += list(df_dict[file_name]["STUDY_ID"])

  # Compute the statistics of features and study_id
  num_file = len(arg_list)
  feature_counter = collections.Counter(feature_list)
  study_id_counter = collections.Counter(study_id_list)
  num_total_features = len(feature_counter)
  num_total_study_id = len(study_id_counter)
  overlapped_features = []
  overlapped_study_id = []
  unique_study_id = []
  for key, counter in feature_counter.iteritems():
    if counter == num_file:
      overlapped_features.append(key)
  for key, counter in study_id_counter.iteritems():
    if counter == num_file:
      overlapped_study_id.append(key)
    elif counter == 1:
      unique_study_id.append(key)

  print "Number of Total features: ", num_total_features
  print "Number of Total patients: ", num_total_study_id
  print "Number of Overlapped features: ", len(overlapped_features)
  print "Number of Overlapped study id: ", len(overlapped_study_id)

  # Re-generate combined data
  keys = df_dict.keys()
  header = []
  f_id = open("test.csv", "wb")
  file_writer = csv.writer(f_id)
  for key in keys:
    for feature in overlapped_features:
      if feature != "STUDY_ID" and feature != "PPTERM":
        if "v1" in key:
          df_dict[key]=df_dict[key].rename(columns = {feature:feature+"V1"})
        if "v2" in key:
          df_dict[key]=df_dict[key].rename(columns = {feature:feature+"V2"}) 

    remove_idx = []
    for study_id in unique_study_id:
      if study_id in list(df_dict[key]['STUDY_ID']):
        remove_idx += df_dict[key].index[df_dict[key]['STUDY_ID'] == study_id].tolist()
    for index in sorted(remove_idx, reverse=True):
      df_dict[key] = df_dict[key].drop(df_dict[key].index[index])

    if key == keys[-1]:
      temp = list(df_dict[key].columns.values)
      header += temp[1:]
    elif key == keys[0]:
      df_dict[key].drop("PPTERM", axis=1, inplace=True)
      header += list(df_dict[key].columns.values)
    else:
      df_dict[key].drop("PPTERM", axis=1, inplace=True)
      temp = list(df_dict[key].columns.values)
      header += temp[1:]
  file_writer.writerow(header)

  for study_id in overlapped_study_id:
    row = []
    for key in df_dict:
      idx = df_dict[key].index[df_dict[key]['STUDY_ID'] == study_id].tolist()
      temp = list(df_dict[key].loc[int(idx[0])])
      for i in range(len(temp)):
        if np.isnan(temp[i]):
          temp[i] = ""
      if key == keys[-1]:
        row += temp[1:]
      elif key == keys[0]:
        row += temp
      else:
        row += temp[1:]
    file_writer.writerow(row)

if __name__ == "__main__":
  main()
