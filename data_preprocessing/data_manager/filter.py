import collections
import re
import pandas as pd
import numpy as np
import time
from googletrans import Translator


regex_logic_equal = re.compile(r"\[(\w+)\]\s?(=)\s?'(\d+)'")
regex_logic_not_equal = re.compile(r"\[(\w+)\]\s?(<>)\s?'(\d+)'")

# All useful tables
table_dict = {'V1':['first_visit', 'med_rec_v1'],
              'V2':['inhome_visit', 'inhome_visit_2nd_part'],
              'V3':['third_visit', 'med_rec_v3', 'food_frequency'],
              'V4':['postpartum_data_abstraction']}
combo_table_list = ['product_use']

# Universal feature names
useless_features = ['FACILITY_ID', 'SYS_LOC_CODE', 'REDCAP_EVENT_NAME', 'EBATCH']
study_id_feature = 'SYS_LOC_CODE'

# def _detect_table():
#   for visit_id in table_dict:
#     if table in table_dict[visit_id]:
#       return visit_id
#
class Data:
  def __init__(self, data_file, data_dictionary, no_ambiguous_data = False):
    self.file_name = data_file
    self.df = pd.read_csv(data_file)
    self.data_columns = self.df.columns.to_list()
    self.data_indices = self.df.index.to_list()
    self.no_ambiguous_data = no_ambiguous_data

    self.categorical_features = []
    self.checkbox_features = []
    self.numerical_features = []

    self.completeness_threshold = 0.8
    self.data_dictionary = data_dictionary

  def filter(self):
    if self.no_ambiguous_data:
      self.df.replace(888, np.nan, inplace=True, regex=True)
      self.df.replace(999, np.nan, inplace=True, regex=True)
    print("Begin filtering...")
    print(self.file_name)
    print("The column number of raw data BEFORE filtered is: ", str(len(self.data_columns)))

    # Drop useless features first
    self.df.drop(useless_features, axis=1, inplace=True)

    # Filter out data columns on feature basis
    removelist_column = []
    for column in self.data_columns:
      if column == study_id_feature:
        continue

      # Remove unchosen features
      if column not in self.data_dictionary.global_mixed_features:
        removelist_column.append(column)
        continue

      null_flags = self.df[column].isnull()
      valid_features_count = collections.Counter(null_flags)[False]
      valid_proportion_by_sample = float(valid_features_count) / float(len(null_flags))
      if valid_proportion_by_sample < self.completeness_threshold:
        removelist_column.append(column)
      else:
        # Find the specific field based on current data
        if column in self.data_dictionary.global_categorical_features:
          self.categorical_features.append(column)
        if column in self.data_dictionary.global_checkbox_features:
          self.checkbox_features.append(column)
        if column in self.data_dictionary.global_numerical_features:
          self.numerical_features.append(column)
    # Drop the columns
    self.df.drop(removelist_column, axis=1, inplace=True)
    # Update the data columns
    self.data_columns = self.df.columns.to_list()

    print("The column number of raw data AFTER filtered is: ", str(len(self.data_columns)))

    # Remove samples having too little valid features
    print("The row number of raw data BEFORE filtered is: " , str(len(self.data_indices)))
    removelist_idx = []
    for i in self.data_indices:
      data_row = self.df.loc[i]
      null_flags = data_row.isnull()
      valid_sample_count = collections.Counter(null_flags)[False]
      valid_proportion_by_feature = float(valid_sample_count) / float(len(null_flags))
      if valid_proportion_by_feature < self.completeness_threshold:
        removelist_idx.append(i)
    # Drop the rows
    self.df.drop(removelist_idx, axis=0, inplace=True)
    # Update the index
    self.df.reset_index(drop=True)
    self.data_indices = self.df.index.to_list()
    print "The row number of raw data AFTER filtered is: " + str(len(data.index))
    print("Filtering finished")


class DataDict:
  def __init__(self, dd_file):
    self.df = pd.read_csv(dd_file)
    self.df = self.df.replace(np.nan, '', regex=True)
    self.dd_columns = self.df.columns.to_list()
    self.dd_indices = self.df.index.to_list()
    self.feature_index = 0
    self.form_index = 1
    self.type_index = 3
    self.description_index = 4
    self.choice_index = 5
    self.text_type_index = 7
    self.branch_logic_index = 11

    self.global_text_features = []
    self.global_categorical_features = []
    self.global_checkbox_features = []
    self.global_numerical_features = []
    self.global_mixed_features = [] # Include both numerical and categorical/checkbox

    self.prefill_val_dict = collections.OrderedDict()

  def extract(self):
    # Human subject data
    print(self.dd_indices)
    for idx in self.dd_indices:
      row = self.df.loc[idx].to_list()
      if row[self.type_index] == "radio" or row[self.type_index] == "dropdown":
        self.global_categorical_features.append(row[self.feature_index].upper())
        self.global_mixed_features.append(row[self.feature_index].upper())
      elif row[self.text_type_index] == "number" or row[self.text_type_index] == "integer":
        self.global_numerical_features.append(row[self.feature_index].upper())
        self.global_mixed_features.append(row[self.feature_index].upper())
      elif row[self.type_index] == "checkbox":
        field_choices = row[self.choice_index]
        sepintlist = field_choices.split('|')
        for item in sepintlist:
          found_int = re.search("\d+", item)
          self.global_checkbox_features.append(row[self.feature_index].upper() + "__" + str(found_int.group()))
          self.global_mixed_features.append(row[self.feature_index].upper() + "__" + str(found_int.group()))
      else:
        self.global_text_features.append(row[self.feature_index].upper())

  def add_prefill_value_for_features_with_deps(self):

    # Translator
    translator = Translator()

    # Human subject data
    for idx in self.dd_indices:
      row = self.df.loc[idx].tolist()
      if row[self.branch_logic_index] == "" or row[self.choice_index] == "":
        continue

      choices = translator.translate(row[self.choice_index]).text.encode('utf-8')
      description = translator.translate(row[self.description_index]).text.encode('utf-8')
      sepintlist = choices.split('|')
      choice_list = []
      for item in sepintlist:
        choice_list.append(int(re.search("(-?\d+),", item).group(1)))
      # Let the google translation server relax
      time.sleep(0.5)

      # Obtain current field specific information
      if row[self.type_index] == "radio" or row[self.type_index] == 'dropdown':
        feature_name = row[self.feature_index].upper()
        if 0 not in choice_list:
          self.prefill_val_dict[feature_name] = 0
        elif "medication" in description.lower() or "medicine" in description.lower():
          self.prefill_val_dict[feature_name] = 0
        elif "characteristics" in description.lower():
          self.prefill_val_dict[feature_name] = 0
        elif "smoke" in description.lower():
          self.prefill_val_dict[feature_name] = 0
        elif 999 in choice_list:
          self.prefill_val_dict[feature_name] = 999

      elif row[self.type_index] == 'checkbox':
        for c in choice_list:
          feature_name = row[self.field_index].upper() + "__" + str(c)
          self.prefill_val_dict[feature_name] = 0
