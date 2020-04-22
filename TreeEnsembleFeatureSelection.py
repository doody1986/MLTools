#! /usr/bin/env python

import sys
import ast
import random
import re
from collections import Counter
import collections
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn import tree
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

import numpy as np
import pandas as pd

import data_preprocessing.data_manager.util

regex_feature_suffix = re.compile(r".*(V\d)$")

##################################################
# data class to hold csv data
##################################################
class data():
  def __init__(self, label_name):
    self.examples = []
    self.features = []
    self.label_name = label_name
    self.label_index = None

##################################################
# function to read in data from the .csv files
##################################################
def read_data(dataset, input_data):
  dataset.examples = input_data[input_data.columns.tolist()].values.tolist()

  #list features
  dataset.features = input_data.columns.tolist()

##################################################
# compute tree
##################################################

# Global parameters
clf = tree.DecisionTreeClassifier(criterion="entropy")

def feat_ranks(dataset):
  data_per_ensemble = [example[:dataset.label_index]+example[dataset.label_index+1:]
                       for example in dataset.examples]
  labels = [example[dataset.label_index] for example in dataset.examples]

  # K fold
  n_fold = 5
  kf = KFold(n_splits=n_fold)

  # Estimator training
  final_accuracy = 0.0
  final_auc = 0.0
  for train_idx, test_idx in kf.split(data_per_ensemble):
    train_set = [data_per_ensemble[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_set = [data_per_ensemble[i] for i in test_idx]
    ref = [labels[i] for i in test_idx]
    clf.fit(train_set, train_labels)
    results = clf.predict(test_set)
    local_accuracy, local_auc = calc_metric(results, ref)
    final_accuracy += local_accuracy
    final_auc += local_auc

  # RFE
  rfe = RFE(clf, 1)
  rfe = rfe.fit(data_per_ensemble, labels)

  return np.array(rfe.ranking_), final_accuracy/float(n_fold), final_auc/float(n_fold)

def calc_metric(results, ref):
  accurate_count = 0
  false_negative_count = 0
  false_positive_count = 0
  true_positive_count = 0
  auc = 0
  fscore = 0
  preterm_count = 0
  term_count = 0
  for i in range(len(results)):
    if results[i] == ref[i]:
      accurate_count += 1
    if ref[i] == 2:
      preterm_count += 1
    if ref[i] == 1:
      term_count += 1
    # False negative
    if ref[i] == 2 and results[i] == 1:
      false_negative_count += 1
    # False positive
    if ref[i] == 1 and results[i] == 2:
      false_positive_count += 1
    # True positive
    if ref[i] == 2 and results[i] == 2:
      true_positive_count += 1
  accuracy = float(accurate_count) / float(len(results))
  false_negative_rate = float(false_negative_count) / float(preterm_count)
  false_positive_rate = float(false_positive_count) / float(term_count)
  true_positive_rate = float(true_positive_count) / float(preterm_count)
  auc = metrics.auc([0.0, false_positive_rate, 1.0], [0.0, true_positive_rate, 1.0])
  return accuracy, auc

# The raw_scores is the output from RFE feature selector
def extract_selected_feat_idx(raw_scores, num_selected_feats):
  # Extract the corresponding index
  scores = [(i, e) for i, e in enumerate(raw_scores)]

  # Sort the above list using the scores
  scores.sort(key=lambda tup: tup[1])

  # Select the corresponding feature indexes
  feature_indexes = [i for i, e in scores[:num_selected_feats]]

  return feature_indexes

##################################################
# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

def Run(input_data, label_name, num_ensemble, method, num_selected_feats,
        missing_rate_table = None, entropy_table = None, manager_datamap = None):

  dataset = data("")
  read_data(dataset, input_data)

  dataset.label_name = label_name

  # The features
  num_feats = len(dataset.features) - 1
  # print "The number of features is: ", num_feats

  #find index of label_name
  for a in range(len(dataset.features)):
    if dataset.features[a] == dataset.label_name:
      dataset.label_index = a
      
  # Split the data set into training and test set
  training_dataset = data(label_name)
  test_dataset = data(label_name)
  training_dataset.features = dataset.features
  test_dataset.features = dataset.features
  for a in range(len(dataset.features)):
    if training_dataset.features[a] == training_dataset.label_name:
      training_dataset.label_index = a
  for a in range(len(dataset.features)):
    if test_dataset.features[a] == test_dataset.label_name:
      test_dataset.label_index = a

  data_samples = dataset.examples
  random.shuffle(data_samples)
  
  negative_samples = filter(lambda x: x[dataset.label_index] == 1, data_samples)
  positive_samples = filter(lambda x: x[dataset.label_index] == 2, data_samples)
  num_negative = len(negative_samples)
  num_positive = len(positive_samples)
  # print "The number of negative sample is: ", num_negative
  # print "The number of positive sample is: ", num_positive

  # No test data is needed during feature selection
  train_idx_positive = range(num_positive)
  train_idx_negative = range(num_negative)

  # Ensemble
  final_feature_list = []
  final_feature_rank = np.zeros(num_feats)
  feature_accuracy_dict = collections.OrderedDict()
  for num in range(num_ensemble):
    new_train_idx_neg = np.random.choice(num_negative, num_positive, replace=False).tolist()
    training_dataset.examples = [ positive_samples[i] for i in train_idx_positive ] +\
                                [ negative_samples[i] for i in new_train_idx_neg ]
    random.shuffle(training_dataset.examples)

    ranks, accuracy, auc = feat_ranks(training_dataset)

    # Feature Selection
    features = dataset.features
    if method == "OFA":
      # Feature occurrence frequency
      feat_idx = extract_selected_feat_idx(ranks, num_selected_feats)
      final_feature_list += [features[i] for i in feat_idx]
    if method == "CLA":
      # Complete linear aggregation
      final_feature_rank += ranks
    if method == "WMA":
      # Weighted mean aggregation
      final_feature_rank += ranks * (1 - auc)
    if method == "CAA":
      # Classification accuracy aggregation
      feat_idx = extract_selected_feat_idx(ranks, num_selected_feats)
      for idx in feat_idx:
        if features[idx] not in feature_accuracy_dict:
          feature_accuracy_dict[features[idx]] = 0
        feature_accuracy_dict[features[idx]] += accuracy
    if method == "MAA":
      # missing rate weighted accuracy aggregation
      feat_idx = extract_selected_feat_idx(ranks, num_selected_feats)
      for idx in feat_idx:
        current_feat = features[idx]
        if current_feat not in feature_accuracy_dict:
          feature_accuracy_dict[current_feat] = 0

        missing_rate = MissingRate(missing_rate_table, current_feat, features)
        alpha = 1.0
        beta = 2.0
        feature_accuracy_dict[current_feat] += accuracy / ((missing_rate + alpha)**beta)
    if method == "EAA":
      # missing rate weighted accuracy aggregation
      feat_idx = extract_selected_feat_idx(ranks, num_selected_feats)
      for idx in feat_idx:
        current_feat = features[idx]
        if current_feat not in feature_accuracy_dict:
          feature_accuracy_dict[current_feat] = 0

        delta_entropy = DeltaEntropy(input_data, entropy_table, current_feat, features)
        alpha = 0.5
        beta = 2.0
        feature_accuracy_dict[current_feat] += accuracy / ((delta_entropy + alpha)**beta)
    if method == "NAA":
      # missing rate weighted accuracy aggregation
      feat_idx = extract_selected_feat_idx(ranks, num_selected_feats)
      for idx in feat_idx:
        current_feat = features[idx]
        if current_feat not in feature_accuracy_dict:
          feature_accuracy_dict[current_feat] = 0

        nmi = NMI(input_data, manager_datamap, current_feat)
        alpha = 1.0
        beta = 2.0
        feature_accuracy_dict[current_feat] += accuracy * nmi
      
  
  if method == "OFA":
    counter_dict = Counter(final_feature_list)
    final_feature_list = sorted(counter_dict, key=counter_dict.get, reverse=True)
  if method == "CLA" or  method == "WMA":
    feat_idx = extract_selected_feat_idx(final_feature_rank, num_selected_feats)
    final_feature_list = [features[i] for i in feat_idx]
  if method == "CAA" or method == "MAA" or method == "EAA" or method == "NAA":
    final_feature_list = sorted(feature_accuracy_dict, key=feature_accuracy_dict.get, reverse=True)

  return final_feature_list[:num_selected_feats]


def main():
  args = str(sys.argv)
  args = ast.literal_eval(args)
  if len(args) < 2:
    print "You have input less than the minimum number of arguments. Go back and read README.txt and do it right next time!"
    exit()
  if args[1][-4:] != ".csv":
    print "Your training file (second argument) must be a .csv!"
    exit()

  input_data = pd.read_csv(args[1])
  method = args[4]
  num_selected_feats = args[5]
  missing_rate_table_path = args[6]
  missing_rate_table = pd.read_csv(missing_rate_table_path)
  entropy_table_path = args[7]
  entropy_table = pd.read_csv(entropy_table_path)
  manager_datamap = args[8]
  print "The feature selection method is: ", method
  final_feature_list = Run(input_data, args[2], int(args[3]), method, int(num_selected_feats),
                           missing_rate_table, entropy_table, manager_datamap)
  print final_feature_list

def MissingRate(missing_rate_table, current_feat, features):
  assert missing_rate_table is not None, "No missing rate table!!!!!!"
  missing_rate_table_features = missing_rate_table.columns.to_list()
  feature_column = missing_rate_table_features[0]
  missing_rate_column = missing_rate_table_features[1]
  visitid_column = missing_rate_table_features[2]
  implicit_visitid = ""
  if regex_feature_suffix.match(current_feat):
    current_feat_raw = current_feat[:-2]
    implicit_visitid = regex_feature_suffix.match(current_feat).group(1)
  else:
    current_feat_raw = current_feat
  missing_rate_list = missing_rate_table[missing_rate_table[
                                      feature_column]==current_feat_raw][missing_rate_column].to_list()
  if len(missing_rate_list) > 1:
    # Multiple visit ID issue
    if implicit_visitid == "":
      visitid_options = ['V1', 'V2', 'V3']
      for opt in visitid_options:
        if current_feat_raw+opt in features:
          continue
        else:
          implicit_visitid = opt
          break

    missing_rate = missing_rate_table.loc[missing_rate_table[
                                      feature_column] == current_feat_raw].loc[missing_rate_table[
                                      visitid_column] == implicit_visitid][missing_rate_column].to_list()[0]
  elif len(missing_rate_list) == 1:
    missing_rate = missing_rate_list[0]

  return missing_rate


def DeltaEntropy(input_data, entropy_table, current_feat, features):
  assert entropy_table is not None, "No entropy table!!!!!!"
  entropy_table_features = entropy_table.columns.to_list()
  feature_column = entropy_table_features[0]
  entropy_column = entropy_table_features[1]
  visitid_column = entropy_table_features[2]
  implicit_visitid = ""
  if regex_feature_suffix.match(current_feat):
    current_feat_raw = current_feat[:-2]
    implicit_visitid = regex_feature_suffix.match(current_feat).group(1)
  else:
    current_feat_raw = current_feat
  entropy_list = entropy_table[entropy_table[
                                      feature_column]==current_feat_raw][entropy_column].to_list()
  if len(entropy_list) > 1:
    # Multiple visit ID issue
    if implicit_visitid == "":
      visitid_options = ['V1', 'V2', 'V3']
      for opt in visitid_options:
        if current_feat_raw+opt in features:
          continue
        else:
          implicit_visitid = opt
          break

    original_entropy = entropy_table.loc[entropy_table[
                                      feature_column] == current_feat_raw].loc[entropy_table[
                                      visitid_column] == implicit_visitid][entropy_column].to_list()[0]
  elif len(entropy_list) == 1:
    original_entropy = entropy_list[0]

  feat_val_list = input_data[current_feat].tolist()
  new_entropy = data_preprocessing.data_manager.util.entropy(feat_val_list)

  return abs(new_entropy - original_entropy)


def NMI(input_data, manager_datamap, current_feat):

  implicit_visitid = ""
  if regex_feature_suffix.match(current_feat):
    current_feat_raw = current_feat[:-2]
    implicit_visitid = regex_feature_suffix.match(current_feat).group(1)
  else:
    current_feat_raw = current_feat

  # Multiple visit ID issue
  if implicit_visitid == "":
    for visitid in manager_datamap:
      for data in manager_datamap[visitid]:
        for feat in data.data_columns:
          if feat != current_feat_raw:
            continue
          feat_vals_before = data.df[feat].tolist()
          feat_vals_before = [val for val in feat_vals_before if not np.isnan(val)]
  else:
    feat_vals_before = manager_datamap[implicit_visitid][current_feat_raw]

  feat_vals_after = input_data[current_feat].tolist()
  nmi = data_preprocessing.data_manager.util.NormMI(feat_vals_before, feat_vals_after, 2)
  print(nmi)
  exit()

  return nmi

if __name__ == "__main__":
  main()
