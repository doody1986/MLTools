#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import time
import re
import collections
from googletrans import Translator

# Only consider these tables so far
table_list = ['first_visit', 'med_rec_v1', 'inhome_visit', 'inhome_visit_2nd_part', 'product_use']

# According the data dictionary
field_index = 0
form_index = 1
type_index = 3
description_index = 4
choice_index = 5
text_type_index = 7
branch_logic_index = 11

regex_logic_equal = re.compile(r"\[(\w+)\]\s?(=)\s?'(\d+)'")
regex_logic_not_equal = re.compile(r"\[(\w+)\]\s?(<>)\s?'(\d+)'")
regex_logic_smaller = re.compile(r"\[(\w+)\]\s?(<)\s?'(\d+)'")

def ExtractFieldsWithDep(dd_file):
  dd = pd.read_csv(dd_file)
  dd = dd.replace(np.nan, '', regex=True)
  writefile = csv.writer(open("fields_with_dep.csv", "wb"))
  header = ["Field Name", "Pre-Filled Value", "Description", "Options", "Depending Logic", "Depending field description", "Depending field option"]
  writefile.writerow(header)
  dd_index = dd.index.tolist()

  # Translator
  translator = Translator()

  # Human subject data
  for idx in dd_index:
    row = dd.loc[idx].tolist()
    if row[form_index] not in table_list:
      continue
    if row[branch_logic_index] == "" or row[choice_index] == "":
      continue

    print "Descrption before translation:", row[description_index]
    print "Options before translation:", row[choice_index]
    description = translator.translate(row[description_index]).text.encode('utf-8')
    print "Descrption after translation:", description
    choices = translator.translate(row[choice_index]).text.encode('utf-8')
    print "Options after translation:", choices
    depending_logic = row[branch_logic_index]


    # Let the google translation server relax
    time.sleep(1)

    # Obtain current field specific information
    if row[type_index] == "radio" or row[type_index] == 'dropdown':
      extracted_row = []
      field_name = row[field_index].upper()

      # Obtain depending field information
      if regex_logic_equal.match(depending_logic) or regex_logic_not_equal.match(depending_logic):
        depending_field = regex_logic_equal.match(depending_logic).group(1) 
        temp_idx = dd.index[dd["Variable / Field Name"]==depending_field].tolist()[0]
        temp_row = dd.loc[temp_idx]
        depending_field_description = translator.translate(temp_row[description_index]).text.encode('utf-8')
        depending_choice = translator.translate(temp_row[choice_index]).text.encode('utf-8')
      else:
        depending_field_description = ""
        depending_choice = ""

      extracted_row = [field_name, "", description, choices, depending_logic, depending_field_description, depending_choice]
      writefile.writerow(extracted_row)
    elif row[type_index] == 'checkbox':
      sepintlist = choices.split('|')
      for item in sepintlist:
        extracted_row = []
        found_int = re.search("(\d+),", item)
        field_name = row[field_index].upper()+"__"+str(found_int.group(1))
        extracted_row = [field_name, "", description, choices, depending_logic, depending_field_description, depending_choice]
        writefile.writerow(extracted_row)

def main():
  print ("Start program.")

  # Extract features into different category
  dd_name = "human_subjects_dd.csv"
  ExtractFieldsWithDep(dd_name)

  print ("End program.")

  return 0

if __name__ == '__main__':
  main()
