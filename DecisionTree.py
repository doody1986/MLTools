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
import pydotplus as pydot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib

##################################################
# data class to hold csv data
##################################################
class data():
  def __init__(self, label_name):
    self.examples = []
    self.balanced_examples = []
    self.features = []
    self.label_name = label_name
    self.label_index = None

##################################################
# function to read in data from the .csv files
##################################################
def read_data(dataset, datafile, datatypes):
  print "Reading CSV data..."
  f = open(datafile)
  original_file = f.read()
  rowsplit_data = original_file.splitlines()
  dataset.examples = [rows.split(',') for rows in rowsplit_data]

  #list features
  dataset.features = dataset.examples.pop(0)
  print "The number of features is: ", len(dataset.features)-1

##################################################
# Preprocess dataset
##################################################
def preprocess(dataset, pos, neg, balanced, multiple_pos=1):
  print "Preprocessing data..."

  negative = neg
  positive = pos
  labels = [example[dataset.label_index] for example in dataset.examples]
  label_counts = Counter(labels)
  num_positive = label_counts[positive]
  majority_label = label_counts.most_common(1)[0][0]

  for feat_index in range(len(dataset.features)):

    neg_samples = filter(lambda x: x[dataset.label_index] == negative, dataset.examples)
    feat_values_neg_samples = [example[feat_index] for example in neg_samples]
             
    pos_samples = filter(lambda x: x[dataset.label_index] == positive, dataset.examples)
    feat_values_pos_samples = [example[feat_index] for example in pos_samples]
        
    feat_values_counts_neg = Counter(feat_values_neg_samples)
    major_feat_value_neg = feat_values_counts_neg.most_common(1)[0][0]
    if major_feat_value_neg == '?' or major_feat_value_neg == '':
      major_feat_value_neg = feat_values_counts_neg.most_common(2)[1][0]

    feat_values_counts_pos = Counter(feat_values_pos_samples)
    major_feat_value_pos = feat_values_counts_pos.most_common(1)[0][0]
    if major_feat_value_pos == '?' or major_feat_value_pos == '':
      major_feat_value_pos = feat_values_counts_pos.most_common(2)[1][0]
    
    for example in dataset.examples:
      if (example[feat_index] == '?' or example[feat_index] == ''):
        if (example[dataset.label_index] == negative):
          example[feat_index] = major_feat_value_neg
        elif (example[dataset.label_index] == positive):
          example[feat_index] = major_feat_value_pos
        else:
          print "No label for this example"
          exit()

    #convert features that are numeric to floats
    for example in dataset.examples:
      example[feat_index] = float(example[feat_index])

  # Get balanced data set
  pos_index_list = []
  neg_index_list = []
  keep_index_list = []
  offset = random.randint(0, len(dataset.examples))
  if balanced == True:
    count = 0
    for i in range(len(dataset.examples)):
      if (dataset.examples[i][dataset.label_index] == float(positive)):
        pos_index_list.append(i)
      j = (i + offset) % len(dataset.examples)
      if (dataset.examples[i][dataset.label_index] == float(negative)) and (count < num_positive * multiple_pos):
        neg_index_list.append(i)
        count += 1
    keep_index_list = pos_index_list + neg_index_list
    dataset.balanced_examples = [ dataset.examples[i] for i in keep_index_list ]


##################################################
# tree node class that will make up the tree
##################################################
class treeNode():
  def __init__(self, is_leaf, classification, feat_split_index, feat_split_value, parent, upper_child, lower_child, height):
    self.is_leaf = True
    self.classification = None
    self.feat_split = None
    self.feat_split_index = None
    self.feat_split_value = None
    self.parent = parent
    self.upper_child = None
    self.lower_child = None
    self.height = None
    self.num_pos = 0
    self.num_neg = 0

##################################################
# compute tree recursively
##################################################

# initialize Tree
  # if dataset is pure (all one result) or there is other stopping criteria then stop
  # for all features a in dataset
    # compute information-theoretic criteria if we split on a
  # abest = best attribute according to above
  # tree = create a decision node that tests abest in the root
  # dv (v=1,2,3,...) = induced sub-datasets from D based on abest
  # for all dv
    # tree = compute_tree(dv)
    # attach tree to the corresponding branch of Tree
  # return tree 

def compute_tree(dataset, parent_node, label_name):
  node = treeNode(True, None, None, None, parent_node, None, None, 0)
  if (parent_node == None):
    node.height = 0
  else:
    node.height = node.parent.height + 1

  num_pos = pos_count(dataset.examples, dataset.features, label_name)
  num_neg = len(dataset.examples) - num_pos
  node.num_pos = num_pos
  node.num_neg = num_neg
  if (len(dataset.examples) == num_pos):
    node.classification = 2
    node.is_leaf = True
    return node
  elif (num_pos == 0):
    node.classification = 1
    node.is_leaf = True
    return node
  else:
    node.is_leaf = False
    if num_pos > num_neg:
      node.classification = 2
    else:
      node.classification = 1
  feat_to_split = None # The index of the attribute we will split on
  max_gain = 0 # The gain given by the best attribute
  split_val = None 
  min_gain = 0.01
  dataset_entropy = calc_dataset_entropy(dataset, label_name)
  dataset_auc = 0.5
  for feat_index in range(len(dataset.features)):
    if (dataset.features[feat_index] != label_name):
      local_max_gain = 0
      local_split_val = None
      # these are the values we can split on, now we must find the best one
      feat_value_list = [example[feat_index] for example in dataset.examples]
      # remove duplicates from list of all attribute values
      feat_value_list = list(set(feat_value_list))
      if(len(feat_value_list) > 100):
        feat_value_list = sorted(feat_value_list)
        total = len(feat_value_list)
        ten_percentile = int(total/10)
        new_list = []
        for x in range(1, 10):
          new_list.append(feat_value_list[x*ten_percentile])
        feat_value_list = new_list

      for val in feat_value_list:
        # calculate the gain if we split on this value
        # if gain is greater than local_max_gain, save this gain and this value
        # calculate the gain if we split on this value
        #local_gain = calc_info_gain(dataset, dataset_entropy, val, feat_index)
        # calculate the gain if we split on this value
        local_gain = calc_gain_auc(dataset, dataset_auc, val, feat_index)
 
        if (local_gain > local_max_gain):
          local_max_gain = local_gain
          local_split_val = val

      if (local_max_gain > max_gain):
        max_gain = local_max_gain
        split_val = local_split_val
        feat_to_split = feat_index

  #feat_to_split is now the best attribute according to our gain metric
  if (split_val is None or feat_to_split is None):
    print "Something went wrong. Couldn't find an attribute to split on or a split value."
  elif (max_gain <= min_gain or node.height > 20):

    node.is_leaf = True
    node.classification = classify_leaf(dataset, label_name)

    return node

  node.feat_split_index = feat_to_split
  node.feat_split = dataset.features[feat_to_split]
  node.feat_split_value = split_val

  # currently doing one split per node so only two datasets are created
  upper_dataset = data(label_name)
  lower_dataset = data(label_name)
  #subset_features = dataset.features[0:node.feat_split_index]+dataset.features[node.feat_split_index+1:len(dataset.features)]
  upper_dataset.features = dataset.features
  lower_dataset.features = dataset.features
  for example in dataset.examples:
    if (feat_to_split is not None and example[feat_to_split] >= split_val):
      upper_dataset.examples.append(example)
    elif (feat_to_split is not None):
      lower_dataset.examples.append(example)

  node.upper_child = compute_tree(upper_dataset, node, label_name)
  node.lower_child = compute_tree(lower_dataset, node, label_name)

  return node

##################################################
# Classify dataset
##################################################
def classify_leaf(dataset, label_name):
  num_pos = pos_count(dataset.examples, dataset.features, label_name)
  total = len(dataset.examples)
  num_neg = total - num_pos
  if (num_pos >= num_neg):
    return 2
  else:
    return 1

##################################################
# Calculate the entropy of the current dataset
##################################################
def calc_dataset_entropy(dataset, label_name):
  num_pos = pos_count(dataset.examples, dataset.features, label_name)
  total_examples = len(dataset.examples);

  entropy = 0
  p = float(num_pos) / float(total_examples)
  if (p != 0):
    entropy += p * math.log(p, 2)
  p = (total_examples - num_pos)/total_examples
  if (p != 0):
    entropy += p * math.log(p, 2)

  entropy = -entropy
  return entropy

##################################################
# Calculate the gain of a particular attribute split
##################################################
def calc_info_gain(dataset, entropy, val, feat_index):
  label_name = dataset.features[feat_index]
  feat_entropy = 0
  total_examples = len(dataset.examples);
  gain_upper_dataset = data(label_name)
  gain_lower_dataset = data(label_name)
  gain_upper_dataset.features = dataset.features
  gain_lower_dataset.features = dataset.features
  for example in dataset.examples:
    if (example[feat_index] >= val):
      gain_upper_dataset.examples.append(example)
    elif (example[feat_index] < val):
      gain_lower_dataset.examples.append(example)

  if (len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0): #Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
    return -1

  feat_entropy += calc_dataset_entropy(gain_upper_dataset, label_name)*float(len(gain_upper_dataset.examples))/float(total_examples)
  feat_entropy += calc_dataset_entropy(gain_lower_dataset, label_name)*float(len(gain_lower_dataset.examples))/float(total_examples)
  return entropy - feat_entropy

##################################################
# Calculate the AUC of a particular attribute split
##################################################
def calc_gain_auc(dataset, auc, val, feat_index):
  label_name = dataset.features[feat_index]
  feat_auc = 0
  total_examples = len(dataset.examples);
  gain_upper_dataset = data(label_name)
  gain_lower_dataset = data(label_name)
  gain_upper_dataset.features = dataset.features
  gain_lower_dataset.features = dataset.features
  for example in dataset.examples:
    if (example[feat_index] >= val):
      gain_upper_dataset.examples.append(example)
    elif (example[feat_index] < val):
      gain_lower_dataset.examples.append(example)

  #Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
  if (len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0):
    return -1

  num_total = len(dataset.examples);
  num_positive = pos_count(dataset.examples, dataset.features, label_name)
  num_negative = num_total - num_positive
  num_total_lower_dataset = len(gain_lower_dataset.examples);
  num_positive_lower_dataset = pos_count(gain_lower_dataset.examples, gain_lower_dataset.features, label_name)
  num_negative_lower_dataset = num_total_lower_dataset - num_positive_lower_dataset
  num_total_upper_dataset = len(gain_upper_dataset.examples);
  num_positive_upper_dataset = pos_count(gain_upper_dataset.examples, gain_upper_dataset.features, label_name)
  num_negative_upper_dataset = num_total_upper_dataset - num_positive_upper_dataset

  lpr_lower = float(num_positive_lower_dataset) / float(num_total_lower_dataset)
  lpr_upper = float(num_positive_upper_dataset) / float(num_total_upper_dataset)
  if lpr_upper > lpr_lower:  
    feat_auc = float(num_positive_upper_dataset * num_negative + num_positive * num_negative_lower_dataset) / float(2 * num_positive * num_negative)
  else:
    feat_auc = float(num_positive_lower_dataset * num_negative + num_positive * num_negative_upper_dataset) / float(2 * num_positive * num_negative)

  return feat_auc - auc

##################################################
# count number of examples with classification "1"
##################################################
def pos_count(instances, features, label_name):
  count = 0
  label_index = None
  #find index of label_name
  for a in range(len(features)):
    if features[a] == label_name:
      label_index = a
    else:
      label_index = len(features) - 1
  for i in instances:
    if i[label_index] == 2:
      count += 1
  return count

##################################################
# Prune tree
##################################################
def prune_tree(root, node, dataset, best_score):
  # if node is a leaf
  if (node.is_leaf == True):
    # get its classification
    classification = node.classification
    # run validate_tree on a tree with the nodes parent as a leaf with its classification
    node.parent.is_leaf = True
    node.parent.classification = node.classification
    if (node.height < 20):
      new_score = validate_tree(root, dataset)
    else:
      new_score = 0
 
    # if its better, change it
    if (new_score >= best_score):
      return new_score
    else:
      node.parent.is_leaf = False
      node.parent.classification = None
      return best_score
  # if its not a leaf
  else:
    # prune tree(node.upper_child)
    new_score = prune_tree(root, node.upper_child, dataset, best_score)
    # if its now a leaf, return
    if (node.is_leaf == True):
      return new_score
    # prune tree(node.lower_child)
    new_score = prune_tree(root, node.lower_child, dataset, new_score)
    # if its now a leaf, return
    if (node.is_leaf == True):
      return new_score

    return new_score

##################################################
# Validate tree
##################################################
def validate_tree(node, dataset):
  total = len(dataset.examples)
  correct = 0
  for example in dataset.examples:
    # validate example
    correct += validate_example(node, example)
  return correct/total

##################################################
# Validate example
##################################################
def validate_example(node, example):
  if (node.is_leaf == True):
    projected = node.classification
    actual = int(example[-1])
    if (projected == actual): 
      return 1
    else:
      return 0
  value = example[node.feat_split_index]
  if (value >= node.feat_split_value):
    return validate_example(node.upper_child, example)
  else:
    return validate_example(node.lower_child, example)

##################################################
# Test example
##################################################
def test_example(example, node):
  if (node.is_leaf == True):
    return node.classification
  else:
    if (example[node.feat_split_index] >= node.feat_split_value):
      return test_example(example, node.upper_child)
    else:
      return test_example(example, node.lower_child)

##################################################
# Print tree
##################################################
def print_tree(node):
  if (node.is_leaf == True):
    for x in range(node.height):
      print "\t",
    print "Classification: " + str(node.classification)
    return
  for x in range(node.height):
      print "\t",
  print "Split index: " + str(node.feat_split)
  for x in range(node.height):
      print "\t",
  print "Split value: " + str(node.feat_split_value)
  print_tree(node.upper_child)
  print_tree(node.lower_child)

##################################################
# Print tree in disjunctive normal form
##################################################
def print_disjunctive(node, dataset, dnf_string):
  if (node.parent == None):
    dnf_string = "( "
  if (node.is_leaf == True):
    if (node.classification == 1):
      dnf_string = dnf_string[:-3]
      dnf_string += ") ^ "
      print dnf_string,
    else:
      return
  else:
    upper = dnf_string + str(dataset.features[node.feat_split_index]) + " >= " + str(node.feat_split_value) + " V "
    print_disjunctive(node.upper_child, dataset, upper)
    lower = dnf_string + str(dataset.features[node.feat_split_index]) + " < " + str(node.feat_split_value) + " V "
    print_disjunctive(node.lower_child, dataset, lower)
    return

##################################################
# Tree node stats
##################################################
def tree_node_stats(node, feature_list, max_height):
  if node.is_leaf == True:
    return
  if node.height == max_height:
    return
  else:
    feature_list.append(node.feat_split)
    tree_node_stats(node.upper_child, feature_list, max_height)
    tree_node_stats(node.lower_child, feature_list, max_height)
    return

##################################################
# Visualize tree
##################################################
POS_NODE_STYLE = {'shape': 'box',
                  'fillcolor': 'blue',
                  'style': 'filled',
                  'fontcolor': 'white'}
NEG_NODE_STYLE = {'shape': 'box',
                  'fillcolor': 'blue',
                  'style': 'filled',
                  'fontcolor': 'white'}
POS_LEAF_STYLE = {'shape': 'ellipse',
                  'fillcolor': '#bd1e24',
                  'style': 'filled',
                  'fontcolor': 'white'}
NEG_LEAF_STYLE = {'shape': 'ellipse',
                  'fillcolor': '#007256',
                  'style': 'filled',
                  'fontcolor': 'white'}

def visualize_tree(root, max_height):
  graph = pydot.Dot('DecisionTree', graph_type='digraph')
  build_tree_graph(graph, root, None, "", max_height)
  graph.write_png('selected_decision_tree.png')

node_id = 0
def build_tree_graph(graph, node, graph_node, edge_label, max_height):
  if node.is_leaf == True or node.height == max_height:
    global node_id
    node_id += 1
    if node.classification == 2:
      leaf_label = "Pre-term"
      #leaf_label = "Malignant"
      leaf = pydot.Node(node_id, label=leaf_label, **POS_LEAF_STYLE)
    elif node.classification == 1:
      leaf_label = "Term"
      #leaf_label = "Benign"
      leaf = pydot.Node(node_id, label=leaf_label, **NEG_LEAF_STYLE)
    graph.add_node(leaf)
    edge = pydot.Edge(graph_node, leaf, label=edge_label)
    graph.add_edge(edge)
  else:
    global node_id
    node_label = node.feat_split+" >= "+str(node.feat_split_value)+"\n"+\
                 "NumPos "+str(node.num_pos)+" NumNeg "+str(node.num_neg)
    node_id += 1
    if node.classification == 2:
      local_graph_node = pydot.Node(node_id, label=node_label, **POS_NODE_STYLE)
    elif node.classification == 1:
      local_graph_node = pydot.Node(node_id, label=node_label, **NEG_NODE_STYLE)
    graph.add_node(local_graph_node)
    if graph_node != None:
      edge = pydot.Edge(graph_node, local_graph_node, label=edge_label)
      graph.add_edge(edge)
    build_tree_graph(graph, node.lower_child, local_graph_node, "No", max_height)
    build_tree_graph(graph, node.upper_child, local_graph_node, "Yes", max_height)

##################################################
# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

def main():
  args = str(sys.argv)
  args = ast.literal_eval(args)
  if (len(args) < 2):
    print "You have input less than the minimum number of arguments. Go back and read README.txt and do it right next time!"
    exit()
  if (args[1][-4:] != ".csv"):
    print "Your training file (second argument) must be a .csv!"
    exit()

  datafile = args[1]
  dataset = data("")
  #if ("-d" in args):
  #  datatypes = args[args.index("-d") + 1]
  #else:
  #  datatypes = 'datatypes.csv'
  # TBD make use of the datatypes
  datatypes = None
  read_data(dataset, datafile, datatypes)
  #arg3 = "PPTERM"
  #arg3 = "Class"
  arg3 = args[2]
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

  balanced = False
  if balanced == True:
    num_preterm = int(args[3])
    print "number of preterm label: ", num_preterm
    preprocess(dataset, '2', '1', balanced, num_preterm)
  preprocess(dataset, '2', '1', balanced)

  data_samples = []
  if balanced == True:
    data_samples = dataset.balanced_examples
  else:
    data_samples = dataset.examples

  random.shuffle(data_samples)
  
  negative_set = filter(lambda x: x[dataset.label_index] == 1, data_samples)
  positive_set = filter(lambda x: x[dataset.label_index] == 2, data_samples)
  print "The number of negative sample is: ", len(negative_set)
  print "The number of positive sample is: ", len(positive_set)

  num_rounds = 10
  final_accuracy = 0.0
  final_false_positive_rate = 0.0
  final_false_negative_rate = 0.0
  final_true_positive_rate = 0.0
  feature_list = []
  selected_tree = None
  ref_false_negative_rate = 1.0
  # Start 10 rounds 10-fold cross validation
  for i in range(num_rounds):
    print "Round "+str(i+1)+" Started"
    n_folds = 10
    cv_arg = KFold(n_folds, shuffle=True)
    root = None
    accuracy = 0.0
    false_negative_rate = 0.0
    false_positive_rate = 0.0
    true_positive_rate = 0.0
    for train_idx, test_idx in cv_arg.split(np.array(data_samples)):
      results = []
      training_dataset.examples = [ data_samples[i] for i in train_idx ]
      test_dataset.examples = [ data_samples[i] for i in test_idx ]
      root = compute_tree(training_dataset, None, label_name)
      tree_node_stats(root, feature_list, 4)
      for example in test_dataset.examples:
        results.append(test_example(example, root))
      ref = [example[test_dataset.label_index] for example in test_dataset.examples]
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
      if preterm_count == 0:
        continue
      accuracy += float(accurate_count) / float(len(results))
      false_negative_rate += float(false_negative_count) / float(preterm_count)
      false_positive_rate += float(false_positive_count) / float(term_count)
      true_positive_rate += float(true_positive_count) / float(preterm_count)
      if (float(false_negative_count) / float(preterm_count)) < ref_false_negative_rate:
        selected_tree = root
        ref_false_negative_rate = float(false_negative_count) / float(preterm_count)

    final_accuracy += accuracy / n_folds
    final_false_negative_rate += false_negative_rate / n_folds
    final_false_positive_rate += false_positive_rate / n_folds
    final_true_positive_rate += true_positive_rate / n_folds

  print "Final accuracy: " + str(final_accuracy / num_rounds)
  print "Final false negative rate: " + str(final_false_negative_rate / num_rounds)
  print "Final false positive rate: " + str(final_false_positive_rate / num_rounds)

  counter_dict = Counter(feature_list)
  print counter_dict
  counter_list = []
  key_list = []
  for pair in counter_dict.most_common():
    key_list.append(pair[0])
    counter_list.append(pair[1])

  # Plot parameter
  opacity = 1.0
  bar_width = 1.
  color_map = [
    '#E8EAF6',
    '#C5CAE9',
    '#9FA8DA',
    '#7986CB',
    '#5C6BC0',
    '#3F51B5',
    '#3949AB',
    '#303F9F',
    '#283593',
    '#1A237E',
  ]

  xtick_name_list = key_list
  n_groups = len(xtick_name_list)

  matplotlib.rc('font', size=20)
  fig = plt.figure(figsize=(15, 10))
  ax = fig.add_subplot(111)

  index = np.arange(n_groups) * bar_width
  ax.bar(index, counter_list, bar_width,
         align='center',
         alpha=opacity,
         color=color_map[0],
         linewidth=3,
         label="Number of Occurence of Selected Features")
 
  ax.set_xlabel('Selected Features')
  ax.set_ylabel('Number of Occurence')
  
  ax.margins(x=0)
  ax.yaxis.grid()
  ax.set_xticks([])
  ##ax.set_xticklabels(layer_list[len(layer_list)-len(gpu_dict[kernel][metric]):])
  #ax.set_xticklabels(xtick_name_list, rotation='45', ha='right')
  #ax.set_xticklabels(xtick_name_list)
  #ax.legend(bbox_to_anchor=(1.05, 1), loc=2,
  #           ncol=1, borderaxespad=0.)
  ax.legend(bbox_to_anchor=(0.35, 1.02, 0.65, .102), loc=3,
            ncol=1, mode="expand", borderaxespad=0.)

  fig.tight_layout()
  fig.savefig('./counter.png', format='png', bbox_inches='tight')
  
  print "Final AUC: ", metrics.auc([0.0, final_false_positive_rate / num_rounds, 1.0], [0.0, final_true_positive_rate/num_rounds, 1.0])

  max_height = 5
  visualize_tree(selected_tree, max_height)

      
if __name__ == "__main__":
  main()
