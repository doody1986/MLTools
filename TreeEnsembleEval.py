#! /usr/bin/env python

import sys
import ast
import random
from collections import Counter
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import tree

import numpy as np
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
  dataset.examples = input_data[input_data.columns.tolist()].values.tolist()

  #list features
  dataset.features = input_data.columns.tolist()

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

def Run(input_data, label_name, num_ensemble, selected_features = []):
  if len(selected_features) != 0:
    selected_feature = selected_features + [label_name]
    input_data = input_data[selected_feature]

  dataset = data("")
  datatypes = None
  read_data(dataset, input_data, datatypes)

  dataset.label_name = label_name

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
  # print results
  # print ref
 
  accurate_count = 0
  false_negative_count = 0
  false_positive_count = 0
  true_positive_count = 0
  auc = 0
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

  # EAA selected features
  # selected_features = ['PPBPADMNSYST', 'MR1WBC', 'PPHISTPHYSC_PULSE', 'MR1FPOSIT', 'PPBPADMNDIAST', 'MR3WBC', 'PPBPPARTUMSYST', 'PPMEMBRANES', 'MR3LYMPHS', 'MR3MCH', 'MR1MONOCTS', 'MR3MCV', 'PPHISTPHYS_BP_DIAST', 'MR1MCHC', 'MR1PLTS', 'MR1MCH', 'MR1CERVLENG', 'MR3MONOCTS', 'MR1GESTAGEWLMP', 'PPPHOTOTHERP']

  # Similarity based missing data handling
  # V1+V2 OFA
  #selected_features = ['MR1WBC', 'MR1FPOSIT', 'MR1RBC', 'MR1MONOCTS', 'MR1MCH', 'MR1PLTS', 'MR1WGHTLBR', 'FVBPSYS', 'FVCHORETIME', 'MR1NEUTRPH', 'FVBPDIAS', 'MR1MCV', 'MR1GESTAGEWLMP', 'MR1FBS', 'PAPTESTMONTH', 'MR1LYMPHS', 'IHV_MOUTHWASH', 'DETERGENTV2', 'WRKCOMMINS', 'IHV_HAIRSPRAY']
  # V1+V2 CLA
  #selected_features = ['MR1WBC', 'MR1FPOSIT', 'PUINTERVIEWERV2', 'HAIRSPRAYV2', 'MAG', 'TAPWATER', 'WRKCOMMINS', 'MULTIFREQ', 'FVBPDIAS', 'IHV_FISH', 'SHAMPOOV2', 'PERFUMEV2', 'POTAS', 'FVURINE', 'FVBLOOD', 'SVHAIR', 'CURCAFUSE', 'OTHERVIT', 'SELEN', 'DEODORV2']
  # V1+V2 WLA
  #selected_features = ['HAIRSPRAYV2', 'GASPUMP', 'IRON', 'PERFUMEV2', 'WRKCOMMINS', 'FOLIC', 'TAPWATER', 'OTHERHAIR2', 'MULTIFREQ', 'VITB12', 'INSTYPE', 'SELEN', 'SVURINE', 'PUINTERVIEWERV2', 'VITE', 'DEODORFFV2', 'CURCAFUSE', 'VITB', 'VITC', 'VITA']
  # V1+V2 CCA
  #selected_features = ['MR1WBC', 'MR1FPOSIT', 'MR1MONOCTS', 'MR1RBC', 'MR1MCH', 'MR1PLTS', 'MR1FBS', 'FVBPSYS', 'IHV_LOTION', 'MR1GESTAGEWLMP', 'IHV_MOUTHWASH', 'HOMTIME', 'MR1NEUTRPH', 'FVCURRHT_INCH', 'FVURINE', 'MR1MCV', 'MR1SPECTYPE', 'MR1FHY', 'FVCURRWT', 'WATFILTER']
  # V1+V2+V3 OFA
  #selected_features = ['MR1WBC', 'MR3MCV', 'CSECTION', 'MR3MCH', 'MR3HCT', 'MR3PLTS', 'MR3RBC', 'MR3WBC', 'MR3LYMPHS', 'MR1FPOSIT', 'MR3NEUTRPH', 'CURRUTI', 'MR1PLTS', 'MR3MONOCTS', 'MR3MCHC', 'FVCURRWT', 'CURRSTD', 'FVPREGSTWT', 'CURRPROM', 'CURRBLD3TRIM']
  # V1+V2+V3 CLA
  #selected_features = ['MR3MCV', 'MR3MCH', 'MR3MONOCTS', 'MR3RBC', 'MR3MCHC', 'MR3HCT', 'MR3WBC', 'MR3PLTS', 'MR3LYMPHS', 'MR3DENSD', 'MR3NEUTRPH', 'MR3PH', 'MR3RESLUC', 'MR3URINALYS', 'MR3UCULT', 'FQBREADTYPE__2', 'FQBREADTYPE__3', 'FQBREADTYPE__4', 'FQBREADTYPE__1', 'MR3CBC']
  # V1+V2+V3 WLA
  #selected_features = ['MR3MCV', 'MR3NEUTRPH', 'MR3HCT', 'MR3MCHC', 'MR3LYMPHS', 'MR3PLTS', 'MR3MONOCTS', 'MR3WBC', 'MR3MCH', 'FQBREADTYPE__3', 'MR3RBC', 'FQBREADTYPE__2', 'MR1WBC', 'FQBREADTYPE__97', 'MR3DENSD', 'MR3CBC', 'FQGREENTYPE__1', 'FQGREENTYPE__2', 'FQBREADTYPE__1', 'FQGREENTYPE__97']
  # V1+V2+V3 CAA
  #selected_features = ['CSECTION', 'MR1WBC', 'MR3LYMPHS', 'MR3NEUTRPH', 'MR3PLTS', 'MR3MCH', 'CURRPROM', 'MR3HCT', 'CURRSTD', 'MR3MCV', 'CURRUTI', 'MR3RBC', 'MR3MCHC', 'MR3WBC', 'MR1FPOSIT', 'CURRBLD3TRIM', 'MR1BPSYST', 'MR1GESTAGEWLMP', 'MR1PLTS', 'MR1FBS']
  # V1+V2+V3+V4 OFA
  #selected_features = ['PPBPADMNSYST', 'MR1WBC', 'PPHISTPHYSC_PULSE', 'PPBPPARTUMSYST', 'PPBPADMNDIAST', 'PPTYPEDEL', 'MR1MONOCTS', 'PPBPDISCHRGDIAST', 'PPBPDSCHRGSYST', 'MR3PLTS', 'MR1RBC', 'MR1GESTAGEWLMP', 'MR1GESTAGEDLMP', 'MR3MONOCTS', 'MR1BPSYST', 'PPBPPARTUM_DIAST', 'MR1FBS', 'MR3MCV', 'PPLABOR', 'MR1FPOSIT']
  # V1+V2+V3+V4 CLA
  #selected_features = ['FQBEEFMIXTYPE__2', 'FQMEXFOODTYPE__5', 'FQPASTASAUCE__8', 'FQMEXFOODTYPE__4', 'FQMEXFOODTYPE__3', 'FQMEXFOODTYPE__1', 'FQBEEFMIXTYPE__1', 'FQCHICKMIXTYPE__1', 'FQMEXFOODTYPE__2', 'FQNUTTYPE__5', 'FQCHICKMIXTYPE__2', 'FQPASTATYPE__2', 'FQBEEFMIXTYPE__3', 'FQNUTTYPE__3', 'FQPASTASAUCE__4', 'FQPASTATYPE__1', 'FQNUTTYPE__2', 'FQNUTTYPE__7', 'FQNUTTYPE__6', 'FQBEEFMIXTYPE__4']
  # V1+V2+V3+V4 WMA
  #selected_features = ['FQPASTASAUCE__4', 'FQPASTATYPE__2', 'FQPASTASAUCE__7', 'FQPASTASAUCE__6', 'FQMEXFOODTYPE__2', 'FQPASTASAUCE__5', 'FQPLANTAINPREP__3', 'FQMEXFOODTYPE__3', 'FQPASTASAUCE__3', 'FQPASTATYPE__3', 'FQMEXFOODTYPE__5', 'FQMEXFOODTYPE__1', 'FQPASTATYPE__1', 'FQPASTASAUCE__8', 'FQADDOIL__97', 'FQBEEFMIXTYPE__2', 'FQMEXFOODTYPE__4', 'FQPASTASAUCE__1', 'FQADDOIL__3', 'FQBEEFMIXTYPE__1']
  # V1+V2+V3+V4 CAA
  #selected_features = ['PPBPADMNSYST', 'MR1WBC', 'MR1GESTAGEWLMP', 'PPBPPARTUM_DIAST', 'MR1FPOSIT', 'MR3MCH', 'PPBPPARTUMSYST', 'MR3PLTS', 'MR3NEUTRPH', 'MR3MONOCTS', 'MR3LYMPHS', 'MR1MONOCTS', 'CSECTION', 'MR3RBC', 'PPMEMBRANES', 'MR3MCHC', 'MR3MCV', 'MR3WBC', 'FVBPSYS', 'MR1PLTS']

  # Average based missing data handling
  # V1+V2 OFA
  #selected_features = ['MR1WBC', 'MR1FPOSIT', 'MR1RBC', 'MR1PLTS', 'MR1MONOCTS', 'BUGSPRY', 'MR1FBS', 'FVBPSYS', 'MR1MCH', 'WRKCOMMINS', 'MR1GESTAGEWLMP', 'MR1LYMPHS', 'FVCURRWT', 'MR1WGHTLBR', 'PREGNUM', 'MR1GESTAGESOND', 'MR1SPECTYPE', 'FVCHORETIME', 'MR1NEUTRPH', 'FVCURRHT_INCH']
  # V1+V2 CLA
  #selected_features = ['WRKCOMMINS', 'MR1WBC', 'MR1FPOSIT', 'INSTYPE', 'TAPWATER', 'GASPUMP', 'MULTIFREQ', 'OTHERVIT', 'FVBLOOD', 'IRON', 'FVURINE', 'DUST', 'VITB6', 'SVHAIR', 'VITB12', 'SVURINE', 'VITE', 'CAL', 'PREVCAFUSE', 'VITD']
  # V1+V2 WLA
  #selected_features = ['VITB6', 'SELEN', 'PUINTERVIEWERV2', 'IHV_FISH', 'TAPWATER', 'VITB12', 'VITE', 'MAG', 'DUST', 'HAIRSPRAYV2', 'SVHAIR', 'POTAS', 'GASPUMP', 'SVURINE', 'VITC', 'FVURINE', 'VISITIDV2', 'OTHERVIT', 'FVBPSYS', 'VITD']
  # V1+V2 CCA
  #selected_features = ['MR1WBC', 'MR1FPOSIT', 'MR1RBC', 'MR1MCH', 'MR1FBS', 'FVBPSYS', 'MR1MONOCTS', 'MR1MCV', 'MR1GESTAGESONW', 'WRKCOMMINS', 'MR1BPDIAST', 'MR1LYMPHS', 'BUGSPRY', 'MR1PLTS', 'FVCURRWT', 'MR1NEUTRPH', 'FVCHORETIME', 'MR1WGHTLBR', 'AGEMENS', 'PREGNUM']
  # V1+V2+V3 OFA
  #selected_features = ['MR1WBC', 'MR3MCV', 'MR3NEUTRPH', 'MR1FPOSIT', 'CSECTION', 'MR3MCH', 'MR3WBC', 'MR3LYMPHS', 'MR3MCHC', 'MR3HCT', 'MR3PLTS', 'MR3DENSD', 'MR3WGHTLBR', 'MR1GESTAGESONW', 'MR1PLTS', 'CURRBLD3TRIM', 'MR1GESTAGEWLMP', 'CURRUTI', 'MR3GESTAGESOND', 'MR3GESTAGESONW']
  # V1+V2+V3 CLA
  #selected_features = ['MR3MCV', 'MR3MCH', 'MR3WBC', 'MR3NEUTRPH', 'MR3LYMPHS', 'MR3PLTS', 'MR3HCT', 'MR3RBC', 'MR3MONOCTS', 'FQBREADTYPE__3', 'FQBREADTYPE__2', 'MR3MCHC', 'MR3UCULT', 'MR3CBC', 'MR3PH', 'FQBREADTYPE__1', 'MR3RESLUC', 'MR3DENSD', 'MR3URINALYS', 'FQBREADTYPE__4']
  # V1+V2+V3 WLA
  #selected_features = ['MR3NEUTRPH', 'MR3WBC', 'MR3MCV', 'MR3LYMPHS', 'MR3PLTS', 'MR3MCHC', 'MR3RBC', 'MR3MCH', 'MR3MONOCTS', 'MR3HCT', 'MR3DENSD', 'MR3CBC', 'MR3RESLUC', 'MR3PH', 'FQBREADTYPE__2', 'MR3UCULT', 'FQBREADTYPE__3', 'MR3URINALYS', 'FQBREADTYPE__1', 'FQGREENTYPE__2']
  # V1+V2+V3 CCA
  #selected_features = ['MR3MCV', 'MR1WBC', 'CSECTION', 'MR3MCH', 'MR3NEUTRPH', 'CURRUTI', 'MR3MCHC', 'MR3RBC', 'CURRBLD3TRIM', 'MR3WBC', 'MR3LYMPHS', 'CURRPROM', 'MR3PLTS', 'CURRBLD2TRIM', 'MR1FBS', 'CURRSTD', 'LIQSOAP', 'MR1FPOSIT', 'PREGNUM', 'MR1PLTS']
  # V1+V2+V3+V4 OFA
  #selected_features = ['PPTYPELABOR', 'MR1WBC', 'PPBPADMNSYST', 'PPBPADMNDIAST', 'PPRECORD_NUMBER', 'PPBPPARTUMSYST', 'PPHISTPHYS_BP_DIAST', 'MR1MONOCTS', 'MR1FBS', 'MR1GESTAGEWLMP', 'PPMEMBRANES', 'MR3MONOCTS', 'MR3PLTS', 'PPBPPARTUM_DIAST', 'PPBPDSCHRGSYST', 'PPHISTPHYSC_PULSE', 'MR3GESTAGESONW', 'MR3NEUTRPH', 'MR1RBC', 'PPHISTPHYS_BP_SYST']
  # V1+V2+V3+V4 CLA
  #selected_features = ['PPTYPELABOR', 'MR1WBC', 'LOTION', 'FQCOFFEETYPE__3', 'FQNUTTYPE__2', 'FQPASTASAUCE__8', 'FQMEXFOODTYPE__4', 'FQPASTASAUCE__7', 'FQSOFTDRINKCAF__3', 'FQCHICKMIXTYPE__2', 'FQBEEFMIXTYPE__2', 'FQSOFTDRINKCAF__1', 'FQSOFTDRINKCAF__2', 'FQMEXFOODTYPE__1', 'FQCHICKMIXTYPE__3', 'FQCHICKMIXTYPE__4', 'FQMEXFOODTYPE__2', 'FQNUTTYPE__3', 'FQMEXFOODTYPE__5', 'FQNUTTYPE__10']
  # V1+V2+V3+V4 WLA
  #selected_features = ['PPTYPELABOR', 'MR1WBC', 'FQCHICKMIXTYPE__4', 'FQCOFFEETYPE__3', 'FQSOFTDRINKCAF__3', 'FQCHICKMIXTYPE__3', 'FQCHICKMIXTYPE__1', 'FQNUTTYPE__1', 'FQSOFTDRINKCAF__1', 'FQCHICKMIXTYPE__2', 'FQSOFTDRINKCAF__97', 'FQNUTTYPE__7', 'FQBEEFMIXTYPE__2', 'FQSOFTDRINKCAF__2', 'FQCOFFEETYPE__1', 'FQNUTTYPE__8', 'FQNUTTYPE__6', 'FQBEEFMIXTYPE__4', 'FQCOFFEETYPE__2', 'FQNUTTYPE__5']
  # V1+V2+V3+V4 CCA
  #selected_features = ['PPTYPELABOR', 'MR1WBC', 'PPBPADMNSYST', 'PPBPPARTUMSYST', 'PPBPADMNDIAST', 'MR1FBS', 'PPRECORD_NUMBER', 'MR1GESTAGEWLMP', 'PPTYPEDEL', 'MR3WBC', 'MR1BPSYST', 'PPMEMBRANES', 'PPHISTPHYSC_PULSE', 'MR1BPDIAST', 'PPHISTPHYS_BP_DIAST', 'FVCURRHT_INCH', 'MR1GESTAGEDLMP', 'MR1RESPR', 'MR3NEUTRPH', 'PPBPDSCHRGSYST']

  #selected_features = ['PPTYPELABOR', 'MR1WBC', 'PAPTESTMONTH', 'PPBPDSCHRGSYST', 'MR1FBS', 'PPBPADMNSYST', 'CSECTION', 'PPBPDISCHRGDIAST', 'PPPHOTOTHERP', 'PPHISTPHYSC_PULSE', 'PPRECORD_NUMBER', 'PPCOMPLIC', 'PREVPREECLMP', 'PPBPADMNDIAST', 'FVBPSYS', 'PREVECLMP', 'FVURINE', 'PPEPISIOT', 'PPTYPEDEL', 'MR3LYMPHS']
  #selected_features = ['PPTYPELABOR', 'MR1WBC', 'PPBPADMNSYST', 'PAPTESTMONTH', 'PPBPDSCHRGSYST', 'PPPHOTOTHERP', 'PPBPADMNDIAST', 'PPRECORD_NUMBER', 'PPBPPARTUMSYST', 'PPTYPEDEL', 'MR1FBS', 'PPCOMPLIC', 'MR1DENSD', 'MR1RBC', 'PPBPDISCHRGDIAST', 'CSECTION', 'PPHISTPHYS_BP_DIAST', 'PPBPPARTUM_DIAST', 'FVBPSYS', 'PPHISTPHYSC_PULSE']
  #selected_features = ['PPTYPELABOR', 'MR1WBC', 'PPBPADMNSYST', 'PPRECORD_NUMBER', 'MR1FBS', 'MR1GESTAGEWLMP', 'PPBPADMNDIAST', 'MR1GESTAGESONW', 'CSECTION', 'PPMEMBRANES', 'PPTYPEDEL', 'MR3PLTS', 'MR3WBC', 'MR3RBC', 'MR3HCT', 'MR1BPSYST', 'PPSTAGETIMING', 'PAPTESTMONTH', 'PPBPPARTUMSYST', 'MR3DENSD']
  #selected_features = ['PPTYPELABOR', 'PPBPADMNSYST', 'MR1WBC', 'PPBPADMNDIAST', 'MR1GESTAGEWLMP', 'PPRECORD_NUMBER', 'PPHISTPHYSC_PULSE', 'MR1FBS', 'PPBPPARTUMSYST', 'MR1GESTAGESONW', 'MR1DENSD', 'MR1BPSYST', 'MR1GESTAGEDLMP', 'PAPTESTMONTH', 'CSECTION', 'PPMEMBRANES', 'FVBPSYS', 'MR1BPDIAST', 'MR1RESLUC', 'MR1LYMPHS']
  selected_features = []
  accuracy, fnr, fpr, auc = Run(input_data, args[2], int(args[3]), selected_features)
  print "Accuracy: ", accuracy
  print "AUC: ", auc


if __name__ == "__main__":
  main()
