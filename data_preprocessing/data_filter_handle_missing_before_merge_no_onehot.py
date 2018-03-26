#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import time
import re
import collections

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
  no_ambiguous_data = False
  if no_ambiguous_data:
    data.replace(888, np.nan, inplace=True, regex=True)
    data.replace(999, np.nan, inplace=True, regex=True)
  print "The column number of raw data BEFORE filtered is: " + str(len(data.columns))
  global categorical_fields
  global checkbox_fields
  global numerical_fields
  categorical_fields = []
  checkbox_fields = []
  numerical_fields = []
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
    if valid_proportion_by_sample > 0.8:
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
    if null_proportion_by_features > 0.2:
      remove_idx.append(i)
  remove_idx.sort(reverse=True)
  for i in remove_idx:
    data = data.drop(data.index[i])
  print "The row number of raw data AFTER filtered is: " + str(len(data.index))

  #data.to_csv("filtered_" + raw_data_file[:-4] + "_" + time.strftime("%m%d%Y") + ".csv")
  return data

def OneHotEncoding(data):
  # One-hot encoding categorical data
  data = pd.get_dummies(data, prefix=categorical_fields, prefix_sep='__', dummy_na=True, columns=categorical_fields)

  # Deal with alread one-hot encoded data
  list_888 = []
  list_999 = []
  for column in checkbox_fields:
    if "888" in column:
      list_888.append(column)
    if "999" in column:
      list_999.append(column)
  if len(list_888) != len(list_999):
    print "Do not make sense!!"
    exit()
  for i in range(len(list_888)):
    data[list_999[i]] = data[list_999[i]] | data[list_888[i]]
    data.drop(list_888[i], axis=1, inplace=True)
  
  return data

def NormalizeNumericalData(data):
  new_data = data.copy()
  if "STUDY_ID" in list(new_data.columns):
    new_data.drop("STUDY_ID", axis=1, inplace=True)
  if "PPTERM" in list(new_data.columns):
    new_data.drop("PPTERM", axis=1, inplace=True)
  for column in numerical_fields:
    mean_ = new_data[column].mean()
    var_ = new_data[column].var()
    new_data[column] = (new_data[column] - mean_) / var_
    min_ = new_data[column].min()
    if min_ < 0:
      new_data[column] = new_data[column] - min_
    max_ = new_data[column].max()
    new_data[column] = new_data[column] / max_
  return new_data

def MissingDataHandling(data):
  # Calculate similarity matrix
  onehot_data = OneHotEncoding(data.copy())
  features = data.columns.tolist()
  indices = data.index.tolist()
  normalized_data = NormalizeNumericalData(onehot_data)

  num_feature = len(features)
  num_sample = len(indices)

  # Initialize the similarity matrix
  similarity_mat = []
  for i in range(num_sample):
    similarity_mat.append([-999999]*num_sample)

  # Do the calculation
  for i in range(num_sample):
    for j in range(i+1, num_sample):
      similarity = 0
      data_i = normalized_data.values[i]
      data_j = normalized_data.values[j]
      product = data_i * data_j
      count = np.count_nonzero(~np.isnan(product))
      similarity = np.nansum(product) / float(count)
      similarity_mat[i][j] = similarity
      similarity_mat[j][i] = similarity
  print "Similarity matrix construction done"

  # Missing data handling
  num_try = 100
  for i in range(num_sample):
    for f in features:
      if np.isnan(data[f][indices[i]]):
        similarity_sample = similarity_mat[i]
        sorted_similarity_sample = sorted(similarity_sample, reverse=True)
        sorted_index = [idx[0] for idx in sorted(zip(indices, similarity_sample), key=lambda x:x[1], reverse=True)]
        for num in range(num_try):
          if np.isnan(data[f][sorted_index[num]]):
            continue
          else:
            print "Attempts:", num, "Selected Similarity:", sorted_similarity_sample[num]
            data.loc[indices[i], f] = data[f][sorted_index[num]]
            break
        if np.isnan(data[f][indices[i]]):
          print "Number of try is not enough"
          exit() 
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

  #num_samples = len(overlapped_study_id)
  #new_index = range(num_samples)
  #for key in data_list:
  #  data_list[key].index = new_index
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

  # Extract features into different category
  dd_name = "human_subjects_dd.csv"
  Extract(dd_name)

  data_list = collections.OrderedDict()
  for file_name in arg_list:
    # Filter the data using 80% completion rate as threshold
    data = Filter(file_name)
    print file_name, "filter done"

    # One hot encoding for the categorical data
    #data = OneHotEncoding(data)
    #print file_name, "one hot encoding done"
    #features = data.columns.tolist()
    #features.remove("PPTERM")
    #data = data[features+["PPTERM"]]

    data = MissingDataHandling(data)
    print "Missing done handling done"

    # Add the data into the data list
    data_list[file_name] = data

  data = Merge(data_list, arg_list)
  print "Merge done"

  data.to_csv("combined_v1_v2_missing_handling_before_merge.csv", index=False)
  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
