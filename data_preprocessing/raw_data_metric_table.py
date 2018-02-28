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

# According the data dictionary
field_index = 0
form_index = 1
type_index = 3
choice_index = 5
text_type_index = 7

global_text_fields = []
global_categorical_fields = []
global_checkbox_fields = []
global_numerical_fields = []
global_mixed_fields = []
def Extract(dd_file):
  readfile = csv.reader(open(dd_file, "r"))

  # Human subject data
  for row in readfile:
    if row[type_index] == "radio" or row[type_index] == 'dropdown':
      global_categorical_fields.append(row[field_index].upper())
      global_mixed_fields.append(row[field_index].upper())
    elif row[text_type_index] == "number" or row[text_type_index] == "integer":
      global_numerical_fields.append(row[field_index].upper())
      global_mixed_fields.append(row[field_index].upper())
    elif row[type_index] == 'checkbox':
      field_choices = row[choice_index]
      sepintlist = field_choices.split('|')
      for item in sepintlist:
        found_int = re.search("\d+", item)
        global_checkbox_fields.append(row[field_index].upper()+"__"+str(found_int.group()))
        global_mixed_fields.append(row[field_index].upper()+"__"+str(found_int.group()))
    else:
      global_text_fields.append(row[field_index].upper())

categorical_fields = []
checkbox_fields = []
numerical_fields = []
def Filter(raw_data_file):
  data = pd.read_csv(raw_data_file)
  no_ambiguous_data = True
  if no_ambiguous_data:
    data.replace(888, np.nan, inplace=True, regex=True)
    data.replace(999, np.nan, inplace=True, regex=True)
  print "The column number of raw data BEFORE filtered is: " + str(len(data.columns))
  global categorical_fields
  global checkbox_fields
  global numerical_fields
  categorical_fields = []
  checkbox_fields = []
  for column in data.columns:
    if column == "STUDY_ID":
      continue
    if column == "PPTERM":
      continue

    # Remove unchosen fields
    if column not in global_mixed_fields:
      data.drop(column, axis=1, inplace=True)
      continue

    null_flags = data[column].isnull()
    remove = True
    valid_fields_count = collections.Counter(null_flags)[False]
    valid_proportion_by_sample = float(valid_fields_count) / float(len(null_flags))
    if valid_proportion_by_sample > 0.5:
      remove = False

    if remove == True:
      data.drop(column, axis=1, inplace=True)
    else:
      # Find the specific field based on current data 
      if column in global_categorical_fields:
        categorical_fields.append(column)
      if column in global_checkbox_fields:
        checkbox_fields.append(column)
      if column in global_numerical_fields:
        numerical_fields.append(column)

  print "The column number of raw data AFTER filtered is: " + str(len(data.columns))

  # Remove some samples with too little valid features
  print "The row number of raw data BEFORE filtered is: " + str(len(data.index))
  remove_idx = []
  for i in data.index:
    data_row = data.iloc[[i]]
    num_features = float(len(data.columns) - 2)
    null_count = 0
    for column in data.columns:
      if column == "STUDY_ID" or column == "PPTERM":
        continue
      if pd.isnull(data_row[column][i]):
        null_count += 1

    null_proportion_by_features = float(null_count) / num_features
    if null_proportion_by_features > 0.5:
      remove_idx.append(i)
  remove_idx.sort(reverse=True)
  for i in remove_idx:
    data = data.drop(data.index[i])
  print "The row number of raw data AFTER filtered is: " + str(len(data.index))

  #data.to_csv("filtered_" + raw_data_file[:-4] + "_" + time.strftime("%m%d%Y") + ".csv")
  return data

def CalcMissingRate(data):
  null_flags = np.isnan(data).tolist()
  null_counter = collections.Counter(null_flags)
  missing_rate = float(null_counter[True]) / float(len(data))
  return "%.4f"%missing_rate

def CalcLinearCorrelation(data, label):
  null_flags = np.isnan(data).tolist()
  valid_idx = [idx for idx, value in enumerate(null_flags) if value == False]
  local_data = data[[valid_idx]]
  local_label = label[[valid_idx]]
  temp = pearsonr(local_data, local_label)
  result = temp[0]
  return "%.4f"%result

def CalcNMI(data, label):
  null_flags = np.isnan(data).tolist()
  valid_idx = [idx for idx, value in enumerate(null_flags) if value == False]
  local_data = data[[valid_idx]]
  local_label = label[[valid_idx]]
  result = normalized_mutual_info_score(local_data, local_label)
  return "%.4f"%result

def CalcAUC_DT(data, label, feature):
  null_flags = np.isnan(data).tolist()
  valid_idx = [idx for idx, value in enumerate(null_flags) if value == False]
  local_data = data[[valid_idx]]
  local_label = label[[valid_idx]]
  num_total = len(local_data)
  label_counter = collections.Counter(local_label)

  # Get the feature value
  feat_value_list = list(set(local_data))
  if(len(feat_value_list) > 100):
    feat_value_list = sorted(feat_value_list)
    total = len(feat_value_list)
    ten_percentile = int(total/10)
    new_list = []
    for x in range(1, 10):
      new_list.append(feat_value_list[x*ten_percentile])
    feat_value_list = new_list

  final_auc = 0.0
  for val in feat_value_list:
    upper_dataset = []
    lower_dataset = []
    upper_label = []
    lower_label = []
    for i in range(len(local_data)):
      if (local_data[i] >= val):
        upper_dataset.append(local_data[i])
        upper_label.append(local_label[i])
      elif (local_data[i] < val):
        lower_dataset.append(local_data[i])
        lower_label.append(local_label[i])
    if (len(upper_dataset) == 0 or len(lower_dataset) == 0):
      continue
    upper_label_counter = collections.Counter(upper_label)
    lower_label_counter = collections.Counter(lower_label)
    num_positive = label_counter[2]
    num_negative = num_total - num_positive
    num_total_lower_dataset = len(lower_dataset);
    num_positive_lower_dataset = lower_label_counter[2]
    num_negative_lower_dataset = num_total_lower_dataset - num_positive_lower_dataset
    num_total_upper_dataset = len(upper_dataset);
    num_positive_upper_dataset = upper_label_counter[2]
    num_negative_upper_dataset = num_total_upper_dataset - num_positive_upper_dataset
    #if "SVHEALTH" in feature:
    #  print "========================"
    #  print "Val:", val
    #  print num_total
    #  print num_positive
    #  print num_negative
    #  print num_total_lower_dataset
    #  print num_positive_lower_dataset
    #  print num_negative_lower_dataset
    #  print num_total_upper_dataset
    #  print num_positive_upper_dataset
    #  print num_negative_upper_dataset

    lpr_lower = float(num_positive_lower_dataset) / float(num_total_lower_dataset)
    lpr_upper = float(num_positive_upper_dataset) / float(num_total_upper_dataset)
    if lpr_upper > lpr_lower:  
      feat_auc = float(num_positive_upper_dataset * num_negative + num_positive * num_negative_lower_dataset) / float(2 * num_positive * num_negative)
    else:
      feat_auc = float(num_positive_lower_dataset * num_negative + num_positive * num_negative_upper_dataset) / float(2 * num_positive * num_negative)
    #if "SVHEALTH" in feature:
    #  print "AUC:", feat_auc
    #  print "========================"

    if feat_auc > final_auc:
      final_auc = feat_auc
  result = final_auc
  #if "SVHEALTH" in feature:
  #  print "Result:", result
  return "%.4f"%result

def CalcAUC_DS(data, label):
  null_flags = np.isnan(data).tolist()
  valid_idx = [idx for idx, value in enumerate(null_flags) if value == False]
  local_data = data[[valid_idx]]
  local_label = label[[valid_idx]]

  # Get the feature value
  feat_value_list = list(set(local_data))
  if(len(feat_value_list) > 100):
    feat_value_list = sorted(feat_value_list)
    total = len(feat_value_list)
    ten_percentile = int(total/10)
    new_list = []
    for x in range(1, 10):
      new_list.append(feat_value_list[x*ten_percentile])
    feat_value_list = new_list

  upper_dataset = []
  lower_dataset = []
  upper_label = []
  lower_label = []
  for val in feat_value_list:
    for i in range(len(local_data)):
      if (local_data[i] >= val):
        upper_dataset.append(local_data[i])
        upper_label.append(local_label[i])
      elif (local_data[i] < val):
        lower_dataset.append(local_data[i])
        lower_label.append(local_label[i])
    if (len(upper_dataset) == 0 or len(lower_dataset) == 0):
      continue

    upper_label_counter = collections.Counter(upper_label)
    lower_label_counter = collections.Counter(lower_label)
  # Unfinished

  return "%.4f"%results

def PrintMetricTable(header, metric_table):
  file_name = "metric_table.csv"
  w = csv.writer(open(file_name, "wb"))
  for row in metric_table:
    w.writerow(row)

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

  # Extract features into different category
  dd_name = "human_subjects_dd.csv"
  Extract(dd_name)

  data_list = collections.OrderedDict()
  header = ["Feature", "Missing Rate", "Linear Correlation", "NMI", "AUC_DT", "AUC_DS"]
  metric_table = []
  metric_table.append(header)
  for file_name in arg_list:
    # Filter the data using 80% completion rate as threshold
    data = Filter(file_name)
    print file_name, "filter done"

    # Get Label
    label = np.array(data['PPTERM'])
    data.drop('PPTERM', axis=1, inplace=True)

    # Remove the Study_id
    data.drop('STUDY_ID', axis=1, inplace=True)

    # For every feature in the data
    features = data.columns.tolist()
    for f in features:
      row = []
      data_f = np.array(data[f])

      if "v1" in file_name:
        feature_name = "V1"+f
      if "v2" in file_name:
        feature_name = "V2"+f

      # Append the feature
      row.append(feature_name)
      # Calculate Missing rate 
      row.append(CalcMissingRate(data_f))
      # Calculate Linear correlation 
      row.append(CalcLinearCorrelation(data_f, label))
      # Calculate NMI
      row.append(CalcNMI(data_f, label))
      # Calculate AUC DT
      row.append(CalcAUC_DT(data_f, label, f))
      # Calculate AUC DS
      #row.append(CalcAUC_DS(data_f, label))
      row.append("")

      metric_table.append(row)
  # Print out the metric table
  #print metric_table
  PrintMetricTable(header, metric_table)

  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
