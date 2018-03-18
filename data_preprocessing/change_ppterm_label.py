#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv

label_list = []
def ObtainLabel(label_file):
  label_data = pd.read_csv(label_file)
  global label_list
  studyids = label_data['studyid'].tolist()
  labels = label_data['preterm_best'].tolist()

  label_list = zip(studyids, [x+1 for x in labels])
   

def UpdateLabel(data_file):
  data = pd.read_csv(data_file)
  for pair in label_list:
    studyid_list = data['STUDY_ID'].tolist()
    if pair[0] in studyid_list:
      idx = data.index[data['STUDY_ID'] == pair[0]].tolist()
      if len(idx) > 1:
        print "Repeated study IDs"
        exit()
      original_value = data.loc[idx[0], 'PPTERM']
      data.loc[idx[0], 'PPTERM'] = pair[1]
      updated_value = data.loc[idx[0], 'PPTERM']
      if original_value != updated_value:
        print "StudyID:", data.loc[idx[0], 'STUDY_ID'], "Original label:", original_value, "Updated label:", updated_value
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
  updated_label_file_name = "postpartum_data.csv"
  ObtainLabel(updated_label_file_name)

  for file_name in arg_list:
    data = UpdateLabel(file_name)
    print file_name, "updated done"

    data.to_csv("updated_"+file_name, index=False)

  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
