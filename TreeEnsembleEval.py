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

def validate_tree(dataset):
  test_set = [example[:dataset.label_index]+example[dataset.label_index+1:] for example in dataset.examples]
  return clf.predict(test_set)

##################################################
# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

def Run(input_data, label_name, num_ensemble):

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
  num_negative = len(negative_samples)
  num_positive = len(positive_samples)
  print "The number of negative sample is: ", num_negative
  print "The number of positive sample is: ", num_positive

  accuracy_list = []
  false_positive_rate_list = []
  false_negative_rate_list = []
  true_positive_rate_list = []
  auc_list = []
  fscore_list = []

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

  predictions = [[] for i in xrange(len(test_dataset.examples))]

  # Ensemble
  for num in range(num_ensemble):
    new_train_idx_neg = np.random.choice(num_neg_training, num_pos_training, replace=False).tolist()
    training_dataset.examples = [ positive_samples[i] for i in train_idx_positive ] +\
                                [ negative_samples[i] for i in new_train_idx_neg ]
    random.shuffle(training_dataset.examples)

    compute_tree(training_dataset)

    preds = validate_tree(test_dataset)
    for i in range(len(preds)):
      predictions[i].append(preds[i])

  # Statistics
  results = []
  for pred_group in predictions:
    counter = Counter(pred_group)
    # Binary class
    # print counter
    assert(len(counter) <= 2)
    if len(counter) == 2:
      if counter[counter.keys()[0]] > counter[counter.keys()[1]]:
        results.append(counter.keys()[0])
      elif counter[counter.keys()[0]] < counter[counter.keys()[1]]:
        results.append(counter.keys()[1])
      elif counter[counter.keys()[0]] == counter[counter.keys()[1]]:
        results.append(1)
    elif len(counter) == 1:
      results.append(counter.keys()[0])
  ref = [example[test_dataset.label_index] for example in test_dataset.examples]
  print results
  print ref
 
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

  return (accuracy, false_negative_rate, false_positive_rate, auc)


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
  # CLA
  #selected_feature = ['MR1GESTAGESONW', 'MR1WGHTLBR', 'MR1BPDIAST', 'FAMCLOSE', 'CHURCHCHNG', 'ACIEVENEGPOS', 'EATCHNGNEGPOS', 'FVURINE', 'RECCHNGNEGPOS', 'FAMCLOSENEGPOS',  'FIREDNEGPOS', 'MR1BPSYST', 'NEWPLCNEGPOS', 'CHURCHCHNGNEGPOS', 'SEP', 'ILL', 'ARGUECHNG', 'ILLNEGPOS', 'FINCHNGNEGPOS', 'WRKCHNGNEGPOS', 'REUNION',     'NEWFAMNEGPOS', 'LIVINGCHNG', 'SEPNEGPOS', 'FIRED', 'MR1WBC', 'WRKCHNG', 'REUNIONNEGPOS', 'SOCCHNG', 'HUSBWRKCHNG']

  # WMA
  #selected_feature = ['MR1WBC', 'EATCHNGNEGPOS', 'SEXDIFF', 'FAMCLOSENEGPOS', 'SEPNEGPOS', 'CHURCHCHNG', 'CHURCHCHNGNEGPOS', 'MR1FHY', 'NEWFAMNEGPOS', 'ARGUECHNGNEGPOS', 'FAMCLOSE', 'NEWPLC', 'REUNION', 'NEWPLCNEGPOS', 'REUNIONNEGPOS', 'SEP', 'FAMILL', 'ARGUECHNG', 'FAMILLNEGPOS', 'FINCHNG', 'NEWJOBNEGPOS', 'FINCHNGNEGPOS', 'NEWJOB', 'DISINFECTANT', 'MR1FPOSIT', 'NEWFAM', 'WRKCHNG', 'HUSBWRKCHNG', 'ACIEVENEGPOS', 'VIOL']

  # OFA
  #selected_feature = ['FVCURRWT', 'FVBPSYS', 'MR1GESTAGESONW', 'MR1WBC', 'FVURINE', 'FVCURRHT_INCH', 'MR1PLTS', 'FVWATDRINK', 'FVBPDIAS', 'MR1FBS', 'SVHEALTH', 'MR1FPOSIT', 'DATEPRENATCAREM', 'MR1MCHC', 'WKSWHENPREG', 'WATSTORE', 'LAUNDRYPROD', 'ALCDAYS', 'LASTALC', 'MR1BPSYST', 'BUGSV2', 'FVWATCOOK', 'IHV_LOTION', 'MR1FIRSTSONG', 'MR1WGHTLBR', 'WATSTOREMAT', 'ALCDRINKS', 'DRUGUSE', 'FVCHORETIME', 'MARIJUSE']

  # CAA
  selected_feature = ['FVBPDIAS', 'MR1WBC', 'MR1MCHC', 'FVINC', 'CSECTION', 'MR1MCH', 'MR1NEUTRPH', 'MR1PLTS', 'SVHEALTH', 'FVCHORETIME', 'MR1RBC', 'FVCURRWT', 'MR1GESTAGESONW', 'ULTRAGESTAGED_FV', 'RACE__3', 'RACE__2', 'RACE__1', 'HISPORG', 'HISP', 'FVMARSTAT', 'RESIDCHANG', 'RACE__4', 'CHILDNUM', 'FVURINE', 'PATED', 'MR1HCT', 'MEDAORALTYPE', 'SMKQUIT', 'LASTCIG', 'SMKDAILY_MY']
  if len(selected_feature) != 0:
    selected_feature = selected_feature + [args[2]]
    print "Evaluate the selected features"
    input_data = input_data[selected_feature]
  accuracy, fnr, fpr, auc = Run(input_data, args[2], int(args[3]))
  print "Accuracy: ", accuracy
  print "AUC: ", auc


if __name__ == "__main__":
  main()
