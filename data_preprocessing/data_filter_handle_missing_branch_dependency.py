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
global_branch_logic_content = collections.OrderedDict()
def Extract(dd_file):
  readfile = csv.reader(open(dd_file, "r"))

  # Human subject data
  for row in readfile:
    if row[form_index] not in table_list:
      continue
    if row[type_index] == "radio" or row[type_index] == 'dropdown':
      field_choices = row[choice_index]
      global_categorical_fields.append(row[field_index].upper())
      global_mixed_fields.append(row[field_index].upper())
      if row[field_index].upper() not in global_branch_logic_content:
        global_branch_logic_content[row[field_index].upper()] = ""
      if row[branch_logic_index] != "":
        global_branch_logic_content[row[field_index].upper()] = row[branch_logic_index]
    elif row[text_type_index] == "number" or row[text_type_index] == "integer":
      global_numerical_fields.append(row[field_index].upper())
      global_mixed_fields.append(row[field_index].upper())
      if row[field_index].upper() not in global_branch_logic_content:
        global_branch_logic_content[row[field_index].upper()] = ""
      if row[branch_logic_index] != "":
        global_branch_logic_content[row[field_index].upper()] = row[branch_logic_index]
    elif row[type_index] == 'checkbox':
      field_choices = row[choice_index]
      sepintlist = field_choices.split('|')
      for item in sepintlist:
        found_int = re.search("(-?\d+),", item)
        global_checkbox_fields.append(row[field_index].upper()+"__"+str(found_int.group(1)))
        global_mixed_fields.append(row[field_index].upper()+"__"+str(found_int.group(1)))
        new_key = row[field_index].upper()+"__"+str(found_int.group(1))
        if row[field_index].upper() not in global_branch_logic_content:
          global_branch_logic_content[new_key] = ""
        if row[branch_logic_index] != "":
          global_branch_logic_content[new_key] = row[branch_logic_index]
    else:
      global_text_fields.append(row[field_index].upper())

regex_logic_equal = re.compile(r"\[(\w+)\]\s?=\s?'?(\d+)'?")
regex_logic_not_equal = re.compile(r"\[(\w+)\]\s?<>\s?'?(\d+)'?")
regex_logic_smaller = re.compile(r"\[(\w+)\]\s?<\s?'?(\d+)'?")
regex_logic_greater = re.compile(r"\[(\w+)\]\s?>\s?'?(\d+)'?")

categorical_fields = []
checkbox_fields = []
numerical_fields = []
def PreFill(raw_data_file, prefill_table):
  data = pd.read_csv(raw_data_file)
  prefill_table = pd.read_csv(prefill_table)
  table_idx = prefill_table.index.tolist()
  field_can_be_prefilled = prefill_table['Field Name'].tolist()
  prefilled_value = prefill_table['Pre-Filled Value'].tolist()
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

    if column in field_can_be_prefilled:
      field_idx = field_can_be_prefilled.index(column)
      field_branch = global_branch_logic_content[column]
      if regex_logic_equal.match(field_branch):
        print "======="
        print "EQUAL"
        print "======="
        print field_branch
        depending_field = regex_logic_equal.match(field_branch).group(1) 
        depending_value = int(regex_logic_equal.match(field_branch).group(2))
        depending_field = depending_field.upper()
        for idx in data_idx:
          if np.isnan(data[column][idx]):
            print column
            print idx
            print data[column][idx]
            print depending_field, depending_value, data[depending_field][idx]
            if data[depending_field][idx] != depending_value:
              data.loc[idx, column] = prefilled_value[field_idx]
              print "Filled:"
              print column, idx
              print "With value", prefilled_value[field_idx]
              print data[column][idx]
      if regex_logic_not_equal.match(field_branch):
        print "======="
        print "NOT EQUAL"
        print "======="
        print field_branch
        depending_field = regex_logic_not_equal.match(field_branch).group(1) 
        depending_value = int(regex_logic_not_equal.match(field_branch).group(2))
        depending_field = depending_field.upper()
        for idx in data_idx:
          if np.isnan(data[column][idx]):
            print column
            print idx
            print data[column][idx]
            print depending_field, depending_value, data[depending_field][idx]
            if data[depending_field][idx] == depending_value:
              data.loc[idx, column] = prefilled_value[field_idx]
              print "Filled:"
              print column, idx
              print "With value", prefilled_value[field_idx]
              print data[column][idx]
      if regex_logic_smaller.match(field_branch):
        print "======="
        print "SMALLER"
        print "======="
        print field_branch
        depending_field = regex_logic_smaller.match(field_branch).group(1) 
        depending_value = int(regex_logic_smaller.match(field_branch).group(2))
        depending_field = depending_field.upper()
        for idx in data_idx:
          if np.isnan(data[column][idx]):
            print column
            print idx
            print data[column][idx]
            print depending_field, depending_value, data[depending_field][idx]
            if data[depending_field][idx] >= depending_value:
              data.loc[idx, column] = prefilled_value[field_idx]
              print "Filled:"
              print column, idx
              print "With value", prefilled_value[field_idx]
              print data[column][idx]
      if regex_logic_greater.match(field_branch):
        print "======="
        print "LARGGER"
        print "======="
        print field_branch
        depending_field = regex_logic_greater.match(field_branch).group(1)
        depending_value = int(regex_logic_greater.match(field_branch).group(2))
        depending_field = depending_field.upper()
        for idx in data_idx:
          if np.isnan(data[column][idx]):
            print column
            print idx
            print data[column][idx]
            print depending_field, depending_value, data[depending_field][idx]
            if data[depending_field][idx] <= depending_value:
              data.loc[idx, column] = prefilled_value[field_idx]
              print "Filled:"
              print column, idx
              print "With value", prefilled_value[field_idx]
              print data[column][idx]
  data.to_csv("prefilled_" + raw_data_file[:-4] + ".csv", index=False)
  return data

def main():
  print ("Start program.")

  if len(sys.argv) < 3:
    print "Too few arguments"
    print "Please specify the data and pre-filled table csv files."
    sys.exit()
  num_args = len(sys.argv)
  if num_args < 3:
    print "There should be at least one input data file!"
    exit()
  file_name = sys.argv[1]
  prefilled_table = sys.argv[2]

  # Extract features into different category
  dd_name = "human_subjects_dd.csv"
  Extract(dd_name)

  data_list = collections.OrderedDict()
  data = PreFill(file_name, prefilled_table)

  return 0

if __name__ == '__main__':
  main()
