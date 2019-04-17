#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import collections

def Filter(raw_data_file, tolerated_missing_rate):
  data = pd.read_csv(raw_data_file)
  no_ambiguous_data = False
  if no_ambiguous_data:
    data.replace(888, np.nan, inplace=True, regex=True)
    data.replace(999, np.nan, inplace=True, regex=True)

  # Remove some samples with too little valid features
  print "The column number of raw data BEFORE filtered is: " + str(len(data.columns))
  for column in data.columns:
    if column == "STUDY_ID":
      continue
    if column == "PPTERM":
      continue

    null_flags = data[column].isnull()
    remove = False
    null_count = collections.Counter(null_flags)[True]
    null_proportion_by_sample = float(null_count) / float(len(null_flags))
    if null_proportion_by_sample > tolerated_missing_rate:
      remove = True

    if remove == True:
      data.drop(column, axis=1, inplace=True)

  print "The column number of raw data AFTER filtered is: " + str(len(data.columns))

  print "The row number of raw data BEFORE filtered is: " + str(len(data.index))
  remove_idx = []
  for i in data.index:
    data_row = data.loc[[i]]
    num_features = float(len(data.columns) - 2)
    null_count = 0
    for column in data.columns:
      if column == "STUDY_ID" or column == "PPTERM":
        continue
      if pd.isnull(data_row[column][i]):
        null_count += 1

    null_proportion_by_features = float(null_count) / num_features
    if null_proportion_by_features > tolerated_missing_rate:
      remove_idx.append(i)
  remove_idx.sort(reverse=True)
  for i in remove_idx:
    data = data.drop(i)
  print "The row number of raw data AFTER filtered is: " + str(len(data.index))

  data.to_csv("filtered_" + raw_data_file[:-4] + "_" + "missing_rate_"+ str(tolerated_missing_rate) + ".csv")
  return data

def Merge(data_list, file_list):
  feature_list = []
  study_id_list = []
  for file_name in file_list:
    features = []
    features = list(data_list[file_name].columns.values)
    indices = data_list[file_name].index.tolist()
    feature_list += features
    study_id_list += list(data_list[file_name]["STUDY_ID"])

  # Compute the statistics of features and study_id
  num_file = len(file_list)
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
  for key in file_list:
    for feature in overlapped_features:
      if feature != "STUDY_ID" and feature != "PPTERM":
        if "v1" in key:
          data_list[key]=data_list[key].rename(columns = {feature:feature+"V1"})
        if "v2" in key:
          data_list[key]=data_list[key].rename(columns = {feature:feature+"V2"}) 

    remove_idx = []
    for study_id in unique_study_id:
      if study_id in list(data_list[key]['STUDY_ID']):
        remove_idx += data_list[key].index[data_list[key]['STUDY_ID'] == study_id].tolist()
    for i in sorted(remove_idx, reverse=True):
      data_list[key] = data_list[key].drop(i)
    overlapped_data = False
    if overlapped_data == True:
      data_list[key].to_csv("overlapped_"+key)
      continue

  for key in file_list:
    if key != file_list[-1]:
      data_list[key].drop("PPTERM", axis=1, inplace=True)
  
  # Merge data
  v1_key = file_list[0]
  v2_key = file_list[1]
  data = data_list[v1_key].merge(data_list[v2_key], on='STUDY_ID')

  # Remove the STUDY_ID column
  data.drop("STUDY_ID", axis=1, inplace=True)

  return data


def main():
  print ("Start program.")

  if len(sys.argv) < 2:
    print "Too few arguments"
    print "Please specify the data csv files."
    sys.exit()
  num_args = len(sys.argv)
  if num_args < 2:
    print "There should be at least one input data file!"
    exit()
  arg_list = sys.argv[1:]

  data_list = collections.OrderedDict()
  for file_name in arg_list:
    # Filter the data using 80% completion rate as threshold
    data = Filter(file_name, 0.0)
    print file_name, "filter done"

    # Add the data into the data list
    data_list[file_name] = data

  data = Merge(data_list, arg_list)
  print "Merge done"

  data.to_csv("combined_v1_v2.csv", index=False)
  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
