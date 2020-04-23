import pandas as pd
import numpy as np
import sys
from scipy.spatial.distance import pdist, squareform

def one_hot_encoding(data, categorical_features, checkbox_features):
  # One-hot encoding categorical data
  data = pd.get_dummies(data, prefix=categorical_features,
                        prefix_sep='__', dummy_na=True, columns=categorical_features)

  # Deal with alread one-hot encoded data
  list_888 = []
  list_999 = []
  for column in checkbox_features:
    if "888" in column:
      list_888.append(column)
    if "999" in column:
      list_999.append(column)
  if len(list_888) != len(list_999):
    print("Do not make sense!!")
    exit()
  for i in range(len(list_888)):
    data[list_999[i]] = data[list_999[i]] | data[list_888[i]]
    data.drop(list_888[i], axis=1, inplace=True)

  return data

def normalize_numerical_data(data, numberical_features, study_id_feature):
  if study_id_feature in list(data.columns):
    data.drop(study_id_feature, axis=1, inplace=True)
  for column in numberical_features:
    mean_ = data[column].mean()
    var_ = data[column].var()
    data[column] = (data[column] - mean_) / var_
    min_ = data[column].min()
    if min_ < 0:
      data[column] = data[column] - min_
    max_ = data[column].max()
    data[column] = data[column] / max_
  return data

def calc_similarity(u, v):
  product = u * v
  count = np.count_nonzero(~np.isnan(product))
  return np.nansum(product) / float(count)

def handler(data, categorical_features, checkbox_features, numberical_features, study_id_feature):
  print("====Missing Data Handling Starts...====")
  # Calculate similarity matrix
  temp_data = data.copy()
  onehot_data = one_hot_encoding(temp_data, categorical_features, checkbox_features)
  features = data.columns.tolist()
  indices = data.index.tolist()
  normalized_data = normalize_numerical_data(onehot_data, numberical_features, study_id_feature)

  num_sample = len(indices)

  # Initialize the similarity matrix
  # similarity_mat = []
  # for i in range(num_sample):
  #   similarity_mat.append([-999999]*num_sample)

  # Do the calculation
  print("Constructing similarity matrix")
  # for i in range(num_sample):
  #   sys.stdout.write('\r>> Progress %.1f%%' % (float(i + 1) / float(num_sample) * 100.0))
  #   sys.stdout.flush()
  #   for j in range(i+1, num_sample):
  #     data_i = normalized_data.values[i]
  #     data_j = normalized_data.values[j]
  #     product = data_i * data_j
  #     count = np.count_nonzero(~np.isnan(product))
  #     similarity = np.nansum(product) / float(count)
  #     similarity_mat[i][j] = similarity
  #     similarity_mat[j][i] = similarity
  similarity_mat = squareform(pdist(normalized_data.values, calc_similarity))
  print("Similarity matrix construction done")

  # Missing data handling
  print("Handling missing data")
  num_try = 150
  for i in range(num_sample):
    # Print the progress
    sys.stdout.write('\r>> Progress %.1f%%' % (float(i + 1) / float(num_sample) * 100.0))
    sys.stdout.flush()
    for f in features:
      if np.isnan(data[f][indices[i]]):
        similarity_sample = similarity_mat[i]
        sorted_similarity_sample = sorted(similarity_sample, reverse=True)
        sorted_index = [idx[0] for idx in sorted(zip(indices, similarity_sample), key=lambda x:x[1], reverse=True)]
        for num in range(num_try):
          if np.isnan(data[f][sorted_index[num]]):
            continue
          else:
            #print "Attempts:", num, "Selected Similarity:", sorted_similarity_sample[num]
            data.loc[indices[i], f] = data[f][sorted_index[num]]
            break
        if np.isnan(data[f][indices[i]]):
          print("Number of try is not enough")
          exit()
  print("\n====Missing Data Handling Finished...====\n")
  return data

def handler1(data, categorical_features, checkbox_features, numberical_features, study_id_feature):
  print("====Missing Data Handling Starts...====")
  features = data.columns.tolist()
  indices = data.index.tolist()
  # Initialize the similarity matrix
  # similarity_mat = []
  # for i in range(num_sample):
  #   similarity_mat.append([-999999]*num_sample)

  # Missing data handling
  print("Handling missing data")
  means = data.mean(axis = 0)

  for f in features:
    # Print the progress
    # sys.stdout.write('\r>> Progress %.1f%%' % (float(i + 1) / float(num_sample) * 100.0))
    # sys.stdout.flush()
    data[f].fillna(means[f], inplace=True)
  print("\n====Missing Data Handling Finished...====\n")
  return data
