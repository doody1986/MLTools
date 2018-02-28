#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import time
import re
import collections

table_list = ['first_visit', 'med_rec_v1', 'inhome_visit', 'inhome_visit_2nd_part', 'product_use']

# According the data dictionary
field_index = 0
form_index = 1
type_index = 3
choice_index = 5
text_type_index = 7
branch_logic_index = 11

global_text_fields = []
global_categorical_fields = []
global_checkbox_fields = []
global_numerical_fields = []
global_mixed_fields = []
global_has_branch_logic = collections.OrderedDict()
global_branch_logic_content = collections.OrderedDict()
global_field_choice_dict = collections.OrderedDict()
def Extract(dd_file):
  readfile = csv.reader(open(dd_file, "r"))

  # Human subject data
  for row in readfile:
    if row[form_index] not in table_list:
      continue
    if row[type_index] == "radio" or row[type_index] == 'dropdown':
      if row[field_index].upper() not in global_field_choice_dict:
        global_field_choice_dict[row[field_index].upper()] = []
      field_choices = row[choice_index]
      sepintlist = field_choices.split('|')
      for item in sepintlist:
        found_int = re.search("(\d+),", item)
        global_field_choice_dict[row[field_index].upper()].append(found_int.group(1))
      global_categorical_fields.append(row[field_index].upper())
      global_mixed_fields.append(row[field_index].upper())
      if row[field_index].upper() not in global_has_branch_logic:
        global_has_branch_logic[row[field_index].upper()] = False
        global_branch_logic_content[row[field_index].upper()] = ""
      if row[branch_logic_index] != "":
        global_has_branch_logic[row[field_index].upper()] = True
        global_branch_logic_content[row[field_index].upper()] = row[branch_logic_index]
    elif row[text_type_index] == "number" or row[text_type_index] == "integer":
      global_numerical_fields.append(row[field_index].upper())
      global_mixed_fields.append(row[field_index].upper())
      if row[field_index].upper() not in global_has_branch_logic:
        global_has_branch_logic[row[field_index].upper()] = False
        global_branch_logic_content[row[field_index].upper()] = ""
      if row[branch_logic_index] != "":
        global_has_branch_logic[row[field_index].upper()] = True
        global_branch_logic_content[row[field_index].upper()] = row[branch_logic_index]
    elif row[type_index] == 'checkbox':
      field_choices = row[choice_index]
      sepintlist = field_choices.split('|')
      for item in sepintlist:
        found_int = re.search("(\d+),", item)
        global_checkbox_fields.append(row[field_index].upper()+"__"+str(found_int.group(1)))
        global_mixed_fields.append(row[field_index].upper()+"__"+str(found_int.group(1)))
        new_key = row[field_index].upper()+"__"+str(found_int.group(1))
        if row[field_index].upper() not in global_has_branch_logic:
          global_has_branch_logic[new_key] = False
          global_branch_logic_content[new_key] = ""
        if row[branch_logic_index] != "":
          print new_key, row[branch_logic_index]
          global_has_branch_logic[new_key] = True
          global_branch_logic_content[new_key] = row[branch_logic_index]
    else:
      global_text_fields.append(row[field_index].upper())

regex_logic_equal = re.compile(r"\[(\w+)\]\s?=\s?'(\d+)'")
regex_logic_not_equal = re.compile(r"\[(\w+)\]\s?<>\s?'(\d+)'")
regex_logic_smaller = re.compile(r"\[(\w+)\]\s?<\s?'(\d+)'")

categorical_fields = []
checkbox_fields = []
numerical_fields = []
def PreFill(raw_data_file):
  data = pd.read_csv(raw_data_file)
  global categorical_fields
  global checkbox_fields
  global numerical_fields
  categorical_fields = []
  checkbox_fields = []
  print "Number of features:", len(data.columns)
  data_idx = data.index.tolist()
  for column in data.columns:
    if column == "STUDY_ID":
      continue
    if column == "PPTERM":
      continue

    # Remove unchosen fields
    if column not in global_mixed_fields:
      data.drop(column, axis=1, inplace=True)
      continue

    if column not in global_has_branch_logic:
      continue

    #print column, global_field_choice_dict[column]
    print column
    if global_has_branch_logic[column]:
      field_branch = global_branch_logic_content[column]
      field_choice = global_field_choice_dict[column]
      if regex_logic_equal.match(column):
        depending_field = regex_logic_equal.match(field_branch).group(1) 
        depending_field = depending_field
        depending_value = regex_logic_equal.match(field_branch).group(2)
        for idx in data_idx:
          if np.isnan(data[column][idx]):
            if data[depending_field][idx] != depending_value:
              data[column][idx] = min(field_choice) - 1
      if regex_logic_not_equal.match(column):
        depending_field = regex_logic_equal.match(field_branch).group(1) 
        depending_field = depending_field
        depending_value = regex_logic_equal.match(field_branch).group(2)
        for idx in data_idx:
          if np.isnan(data[column][idx]):
            if data[depending_field][idx] == depending_value:
              data[column][idx] = min(field_choice) - 1
  data.to_csv("prefilled_" + raw_data_file[:-4] + ".csv")
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
    data = PreFill(file_name)

  return 0

if __name__ == '__main__':
  main()
