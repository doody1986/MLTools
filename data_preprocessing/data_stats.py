#! /usr/bin/env python
import sys 
import numpy as np
import pandas as pd
import collections
import csv
import math

def main():
  num_args = len(sys.argv)
  if num_args < 2:
    print "There should be at least one input data file!"
    exit()
  arg_list = sys.argv[1:]
  feature_list = []
  study_id_list = []
  df_dict = collections.OrderedDict()

  no_ambiguous_data = True
  # Get the dataframe
  for file_name in arg_list:
    features = []
    df_dict[file_name] = pd.read_csv(file_name)
    if no_ambiguous_data:
      df_dict[file_name].replace('888', np.nan, inplace=True, regex=True)
      df_dict[file_name].replace('999', np.nan, inplace=True, regex=True)
    features = list(df_dict[file_name].columns.values)
    indices = df_dict[file_name].index.tolist()
    feature_list += features
    study_id_list += list(df_dict[file_name]["STUDY_ID"])

    ## Preliminary missing data handling
    #pos_index = df_dict[file_name].index[df_dict[file_name]['PPTERM'] == 2].tolist()
    #neg_index = df_dict[file_name].index[df_dict[file_name]['PPTERM'] == 1].tolist()
    #for feat in features:
    #  null_flags = list(df_dict[file_name][feat].isnull())
    #  if any(null_flags):
    #    pos_samples = list(df_dict[file_name][feat][pos_index])
    #    neg_samples = list(df_dict[file_name][feat][neg_index])
    #    feat_values_counts_pos = collections.Counter(pos_samples)
    #    major_feat_value_pos = feat_values_counts_pos.most_common(1)[0][0]
    #    feat_values_counts_neg = collections.Counter(neg_samples)
    #    major_feat_value_neg = feat_values_counts_neg.most_common(1)[0][0]
    #    for i in pos_index:
    #      if np.isnan(df_dict[file_name][feat][i]):
    #        df_dict[file_name].at[i, feat] = major_feat_value_pos
    #    for i in neg_index:
    #      if np.isnan(df_dict[file_name][feat][i]):
    #        df_dict[file_name].at[i, feat] = major_feat_value_neg
    #pos_index = df_dict[file_name].index[df_dict[file_name]['PPTERM'] == 2].tolist()
    #neg_index = df_dict[file_name].index[df_dict[file_name]['PPTERM'] == 1].tolist()
    #all_index = df_dict[file_name].index.tolist()
    #for feat in features:
    #  null_flags = list(df_dict[file_name][feat].isnull())
    #  if any(null_flags):
    #    samples = list(df_dict[file_name][feat])
    #    feat_values_counts = collections.Counter(samples)
    #    major_feat_value = feat_values_counts.most_common(1)[0][0]
    #    for i in all_index:
    #      if np.isnan(df_dict[file_name][feat][i]):
    #        #df_dict[file_name].at[i, feat] = major_feat_value
    #        df_dict[file_name].at[i, feat] = 555

    # Missing data handling using similarity
    num_sample = len(indices)
    num_feature = len(features)

    # Initialize the similarity matrix
    similarity_mat = []
    for i in range(num_sample):
      similarity_mat.append([-999999]*num_sample)

    # Do the calculation
    for i in range(num_sample):
      data_i = list(df_dict[file_name].loc[int(indices[i])])
      for j in range(i+1, num_sample):
        similarity = 0
        count = 0
        data_j = list(df_dict[file_name].loc[int(indices[j])])
        for f in range(num_feature):
          if np.isnan(data_i[f]) or np.isnan(data_j[f]):
            continue
          else:
            similarity -= (float(data_i[f]) - float(data_j[f])) * (float(data_i[f]) - float(data_j[f]))
            count += 1
        similarity_mat[i][j] = similarity# / float(count)
        similarity_mat[j][i] = similarity# / float(count)
    print "Similarity matrix construction done"

    # Missing data handling
    num_try = 100
    for i in range(num_sample):
      data_i = list(df_dict[file_name].loc[int(indices[i])])
      for f in range(num_feature):
        if np.isnan(data_i[f]):
          similarity_sample = similarity_mat[i]
          sorted_similarity_sample = sorted(similarity_sample, reverse=True)
          sorted_index = [index[0] for index in sorted(enumerate(similarity_sample), key=lambda x:x[1], reverse=True)]
          for num in range(num_try):
            data_x = list(df_dict[file_name].loc[int(indices[sorted_index[num]])])
            if np.isnan(data_x[f]):
              continue
            else:
              df_dict[file_name].at[features[f], i] = data_x[f]
              print "Feature:", f
              print "Selected Similarity:", sorted_similarity_sample[sorted_index[num]]
              break
          if np.isnan(data_i[f]):
            print "Number of try is not enough"
            exit()

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
  fd = open("combined_v1_v2.csv", "wb")
  file_writer = csv.writer(fd)
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
    overlapped_data = False
    if overlapped_data == True:
      df_dict[key].to_csv("overlapped_"+key)
      continue

    if key == keys[-1]:
      temp = list(df_dict[key].columns.values)
      header += temp[1:]
    elif key == keys[0]:
      df_dict[key].drop("PPTERM", axis=1, inplace=True)
      temp = list(df_dict[key].columns.values)
      header += temp[1:]
    else:
      df_dict[key].drop("PPTERM", axis=1, inplace=True)
      temp = list(df_dict[key].columns.values)
      header += temp[1:]

  file_writer.writerow(header)

  #numerical_field_indicator = open("numerical_chosenfields.csv", "rb")
  #file_reader = csv.reader(numerical_field_indicator)
  #numerical_fields_list = []
  #numerical_fields_index_list = []
  #for field in file_reader:
  #  if field[0] in header:
  #    numerical_fields_list.append(field[0])
  #    numerical_fields_index_list.append(header.index(field[0]))
  #print len(numerical_fields_list), "features for calculating similarity matrix"

  # Get raw data
  data = []
  for study_id in overlapped_study_id:
    row = []
    for key in df_dict:
      idx = df_dict[key].index[df_dict[key]['STUDY_ID'] == study_id].tolist()
      temp = list(df_dict[key].loc[int(idx[0])])
      for i in range(len(temp)):
        if np.isnan(temp[i]):
          print "Something is wrong"
          exit()
      if key == keys[-1]:
        row += temp[1:]
      elif key == keys[0]:
        row += temp[1:]
      else:
        row += temp[1:]
    data.append(row)

  ##
  # Calculate similarity matrix
  ##
  #sub_data = []
  #for sample in data:
  #  sub_data.append([sample[i] for i in numerical_fields_index_list])

  #num_sample = len(data)
  #num_feature = len(data[0])
  #num_sub_feature = len(sub_data[0])

  ## Initialize the similarity matrix
  #similarity_mat = []
  #for i in range(num_sample):
  #  similarity_mat.append([-999999]*num_sample)

  ## Do the calculation
  #for i in range(num_sample):
  #  for j in range(i+1, num_sample):
  #    similarity = 0
  #    count = 0
  #    for f in range(num_sub_feature):
  #      if data[i][f] == "" or data[j][f] == "":
  #        continue
  #      else:
  #        similarity -= float(data[i][f] - data[j][f]) * float(data[i][f] - data[j][f])
  #        count += 1
  #    similarity_mat[i][j] = similarity / float(count)
  #    similarity_mat[j][i] = similarity / float(count)
  #print "Similarity matrix construction done"

  ## Missing data handling
  #num_try = 100
  #for i in range(num_sample):
  #  for f in range(num_feature):
  #    if data[i][f] == "":
  #      similarity_sample = similarity_mat[i]
  #      sorted_similarity_sample = sorted(similarity_sample, reverse=True)
  #      sorted_index = [index[0] for index in sorted(enumerate(similarity_sample), key=lambda x:x[1], reverse=True)]
  #      for num in range(num_try):
  #        if data[sorted_index[num]][f] == "":
  #          continue
  #        else:
  #          data[i][f] = data[sorted_index[num]][f]
  #          print "Feature:", header[f]
  #          print "Selected Similarity:", sorted_similarity_sample[sorted_index[num]]
  #          break
  #      if data[i][f] == "":
  #        print "Number of try is not enough"
  #        exit()

  for row in data:
    file_writer.writerow(row)

if __name__ == "__main__":
  main()
