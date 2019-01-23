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
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import tree

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
  print "The number of features is: ", len(dataset.features)-1

##################################################
# compute tree
##################################################

clf = tree.DecisionTreeClassifier(criterion="entropy")

def compute_tree(dataset):
  train_set = [example[:dataset.label_index]+example[dataset.label_index+1:] for example in dataset.examples]
  train_label = [example[dataset.label_index] for example in dataset.examples]
  clf.fit(train_set, train_label)
  print clf.feature_importances_
  print len(clf.feature_importances_)
  exit()

def validate_tree(dataset):
  test_set = [example[:dataset.label_index]+example[dataset.label_index+1:] for example in dataset.examples]
  return clf.predict(test_set)

##################################################
# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

def Run(input_data, label_name):

  dataset = data("")
  datatypes = None
  read_data(dataset, input_data, datatypes)
  arg3 = label_name
  if (arg3 in dataset.features):
    label_name = arg3
  else:
    label_name = dataset.features[-1]

  dataset.label_name = label_name

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
  print "The number of negative sample is: ", len(negative_samples)
  print "The number of positive sample is: ", len(positive_samples)

  accuracy_list = []
  false_positive_rate_list = []
  false_negative_rate_list = []
  true_positive_rate_list = []
  auc_list = []
  fscore_list = []
  #ref_false_negative_rate = 1.0
  # Start 10-fold cross validation
  n_folds = 10
  cv_arg = KFold(n_folds, shuffle=True)
  root = None
  #avg_num_feature = 0
  #feature_list = []
  #selected_tree = None

  train_idx_positive = []
  test_idx_positive = []
  train_idx_negative = []
  test_idx_negative = []
  for train_idx, test_idx in cv_arg.split(np.array(positive_samples)):
    train_idx_positive.append(train_idx)
    test_idx_positive.append(test_idx)
  for train_idx, test_idx in cv_arg.split(np.array(negative_samples)):
    train_idx_negative.append(train_idx)
    test_idx_negative.append(test_idx)

  accuracy = 0.0
  false_negative_rate = 0.0
  false_positive_rate = 0.0
  true_positive_rate = 0.0
  auc = 0.0
  fscore = 0.0
  for idx in range(n_folds):
    training_dataset.examples = [ positive_samples[i] for i in train_idx_positive[idx] ] +\
                                [ negative_samples[i] for i in train_idx_negative[idx] ]
    test_dataset.examples = [ positive_samples[i] for i in test_idx_positive[idx] ] +\
                            [ negative_samples[i] for i in test_idx_negative[idx] ]
    random.shuffle(training_dataset.examples)
    random.shuffle(test_dataset.examples)
    compute_tree(training_dataset)

    ref = [example[test_dataset.label_index] for example in test_dataset.examples]
    
    results = validate_tree(test_dataset)
    accurate_count = 0
    false_negative_count = 0
    false_positive_count = 0
    true_positive_count = 0
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
    accuracy += float(accurate_count) / float(len(results))
    local_false_negative_rate = float(false_negative_count) / float(preterm_count)
    false_negative_rate += local_false_negative_rate
    local_false_positive_rate = float(false_positive_count) / float(term_count)
    false_positive_rate += local_false_positive_rate
    local_true_positive_rate = float(true_positive_count) / float(preterm_count)
    true_positive_rate += local_true_positive_rate
    localauc = metrics.auc([0.0, local_false_positive_rate, 1.0], [0.0, local_true_positive_rate, 1.0])
    auc += localauc
    fscore += metrics.fbeta_score(ref, results, 2)

  accuracy = accuracy / n_folds
  false_negative_rate = false_negative_rate / n_folds
  false_positive_rate = false_positive_rate / n_folds
  true_positive_rate = true_positive_rate / n_folds
  auc = auc / n_folds
  fscore = fscore / n_folds
    #if (float(false_negative_count) / float(preterm_count)) < ref_false_negative_rate:
    #  selected_tree = root
    #  ref_false_negative_rate = float(false_negative_count) / float(preterm_count)

  return (accuracy, false_negative_rate, false_positive_rate, auc, fscore)
  
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
  accuracy, fnr, fpr, auc, fscore = Run(input_data, args[2])
  print accuracy
  print auc


if __name__ == "__main__":
  main()
