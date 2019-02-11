#! /usr/bin/env python

import math
import operator
import time
import random
import copy
import sys
import ast
import csv
import random
from collections import Counter
import collections
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import tree
from sklearn.feature_selection import RFE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib

import pandas as pd

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
def read_data(dataset, input_data, datatypes):
  dataset.examples = input_data[input_data.columns.tolist()].as_matrix().tolist()

  #list features
  dataset.features = input_data.columns.tolist()

##################################################
# compute tree
##################################################

# Global parameters
clf = tree.DecisionTreeClassifier(criterion="entropy")
def feat_ranks(dataset):
  train_set = [example[:dataset.label_index]+example[dataset.label_index+1:] for example in dataset.examples]
  train_label = [example[dataset.label_index] for example in dataset.examples]

  # Estimator training
  clf.fit(train_set, train_label)

  # RFE
  rfe = RFE(clf, 1)
  rfe = rfe.fit(train_set, train_label)
  return np.array(rfe.ranking_)

def validate(dataset):
  test_set = [example[:dataset.label_index]+example[dataset.label_index+1:] for example in dataset.examples]
  return clf.predict(test_set)

# The raw_scores is the output from RFE feature selector
def extract_selected_feat_idx(raw_scores, num_selected_feats):
  # Extract the corresponding index
  scores = [(i, e) for i, e in enumerate(raw_scores)]

  # Sort the above list using the scores
  scores.sort(key=lambda tup: tup[1])

  # Select the corresponding feature indexes
  feature_indexes = [i for i, e in scores[:num_selected_feats]]

  return feature_indexes

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

##################################################
# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

def Run(input_data, label_name, num_ensemble, method, num_selected_feats):

  dataset = data("")
  datatypes = None
  read_data(dataset, input_data, datatypes)
  arg3 = label_name
  if (arg3 in dataset.features):
    label_name = arg3
  else:
    label_name = dataset.features[-1]

  dataset.label_name = label_name

  # The features
  features = dataset.features[:-1]
  num_feats = len(features)
  print "The number of features is: ", num_feats

  #find index of label_name
  for a in range(len(dataset.features)):
    if dataset.features[a] == dataset.label_name:
      dataset.label_index = a
    else:
      dataset.label_index = range(len(dataset.features))[-1]
      
  # Split the data set into training and test set
  training_dataset = data(label_name)
  test_dataset = data(label_name)
  training_dataset.features = dataset.features
  test_dataset.features = dataset.features
  for a in range(len(dataset.features)):
    if training_dataset.features[a] == training_dataset.label_name:
      training_dataset.label_index = a
    else:
      training_dataset.label_index = range(len(training_dataset.features))[-1]
  for a in range(len(dataset.features)):
    if test_dataset.features[a] == test_dataset.label_name:
      test_dataset.label_index = a
    else:
      test_dataset.label_index = range(len(test_dataset.features))[-1]

  data_samples = dataset.examples
  random.shuffle(data_samples)
  
  negative_samples = filter(lambda x: x[dataset.label_index] == 1, data_samples)
  positive_samples = filter(lambda x: x[dataset.label_index] == 2, data_samples)
  num_negative = len(negative_samples)
  num_positive = len(positive_samples)
  print "The number of negative sample is: ", num_negative
  print "The number of positive sample is: ", num_positive

  test_propotion = 0.1

  train_idx_positive = range(num_positive)
  test_idx_positive = []
  train_idx_negative = range(num_negative)
  test_idx_negative = []
  num_test_pos = int(round(num_positive * test_propotion))
  num_test_neg = int(round(num_negative * test_propotion))
  test_idx_positive = np.random.choice(num_positive, num_test_pos, replace=False).tolist()
  test_idx_negative = np.random.choice(num_negative, num_test_neg, replace=False).tolist()
  train_idx_positive = filter(lambda x: x not in test_idx_positive, train_idx_positive)
  train_idx_negative = filter(lambda x: x not in test_idx_negative, train_idx_negative)
  num_pos_training = len(train_idx_positive)
  num_neg_training = len(train_idx_negative)

  test_dataset.examples = [ positive_samples[i] for i in test_idx_positive ] +\
                          [ negative_samples[i] for i in test_idx_negative ]
  random.shuffle(test_dataset.examples)

  # Ensemble
  final_feature_list = []
  final_feature_rank = np.zeros(num_feats)
  ref = [example[test_dataset.label_index] for example in test_dataset.examples]
  for num in range(num_ensemble):
    new_train_idx_neg = np.random.choice(num_neg_training, num_pos_training, replace=False).tolist()
    training_dataset.examples = [ positive_samples[i] for i in train_idx_positive ] +\
                                [ negative_samples[i] for i in new_train_idx_neg ]
    random.shuffle(training_dataset.examples)

    ranks = feat_ranks(training_dataset)

    results = validate(test_dataset)
    accuracy, auc = calc_metric(results, ref)

    # Feature Selection
    if method == "OFA":
      # Feature occurrence frequency
      feat_idx = extract_selected_feat_idx(ranks, num_selected_feats)
      final_feature_list += [features[i] for i in feat_idx]
    if method == "CLA":
      # Complete linear aggregation
      final_feature_rank += ranks
    if method == "WMA":
      # 
      final_feature_rank += ranks * (1 - auc)
  
  if method == "OFA":
    counter_dict = Counter(final_feature_list)
    final_feature_list = sorted(counter_dict, key=counter_dict.get, reverse=True)
  if method == "CLA" or  method == "WMA":
    feat_idx = extract_selected_feat_idx(final_feature_rank, num_selected_feats)
    final_feature_list = [features[i] for i in feat_idx]

  return final_feature_list[:num_selected_feats]


def main():
  args = str(sys.argv)
  args = ast.literal_eval(args)
  if (len(args) < 2):
    print "You have input less than the minimum number of arguments. Go back and read README.txt and do it right next time!"
    exit()
  if (args[1][-4:] != ".csv"):
    print "Your training file (second argument) must be a .csv!"
    exit()

  input_data = pd.read_csv(args[1])
  method = "WMA"
  print "The feature selection method is: ", method
  final_feature_list = Run(input_data, args[2], int(args[3]), method, 20)
  print final_feature_list


if __name__ == "__main__":
  main()
