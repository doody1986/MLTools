import collections
import re
import pandas as pd
import numpy as np
import time
from googletrans import Translator


regex_logic_equal = re.compile(r"\[(\w+)\]\s?(=)\s?'(\d+)'")
regex_logic_not_equal = re.compile(r"\[(\w+)\]\s?(<>)\s?'(\d+)'")

# Need special care
# Need to add suffix to each features using visit ID
combo_table_list = ['product_use']

# Universal feature names
useless_features = ['FACILITY_ID', 'SYS_LOC_CODE', 'REDCAP_EVENT_NAME', 'EBATCH']
study_id_feature = 'SYS_LOC_CODE'


class Data:
  def __init__(self, data_file, data_dictionary, no_ambiguous_data = False):
    self.file_name = data_file
    self.df = pd.read_csv(data_file, low_memory=False)
    # Directly drop the useless features from SQL server
    self.df.drop(useless_features, axis=1, inplace=True)
    self.data_columns = self.df.columns.to_list()
    self.data_indices = self.df.index.to_list()
    self.no_ambiguous_data = no_ambiguous_data

    self.categorical_features = []
    self.checkbox_features = []
    self.numerical_features = []

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




class Manager:
  def __init__(self, data_path_map, data_dictionary_path):
    # Extract data dictionary
    self.data_dictionary = DataDict(data_dictionary_path)
    self.extract_dd()
    
    self.data_map = {}
    # Get data
    for visitid in data_path_map:
      if visitid not in self.data_map:
        self.data_map[visitid] = []
      for data_file in data_path_map[visitid]:
        data = Data(data_file, self.data_dictionary)
        self.data_map[visitid].append(data)

    self.completeness_threshold = 0.8
    self.prefill_val_dict = collections.OrderedDict()

  def extract_dd(self):
    # Human subject data
    dd = self.data_dictionary
    for idx in dd.dd_indices:
      row = dd.df.loc[idx].to_list()
      if row[dd.type_index] == "radio" or row[dd.type_index] == "dropdown":
        dd.global_categorical_features.append(row[dd.feature_index].upper())
        dd.global_mixed_features.append(row[dd.feature_index].upper())
      elif row[dd.text_type_index] == "number" or row[dd.text_type_index] == "integer":
        dd.global_numerical_features.append(row[dd.feature_index].upper())
        dd.global_mixed_features.append(row[dd.feature_index].upper())
      elif row[dd.type_index] == "checkbox":
        field_choices = row[dd.choice_index]
        sepintlist = field_choices.split('|')
        for item in sepintlist:
          found_int = re.search("\d+", item)
          dd.global_checkbox_features.append(row[dd.feature_index].upper() + "__" + str(found_int.group()))
          dd.global_mixed_features.append(row[dd.feature_index].upper() + "__" + str(found_int.group()))
      else:
        dd.global_text_features.append(row[dd.feature_index].upper())

  def add_prefill_value_for_features_with_deps(self):
    # Need further action to make use of files
    dd = self.data_dicationary
    # Translator
    translator = Translator()

    # Human subject data
    for idx in dd.dd_indices:
      row = dd.df.loc[idx].tolist()
      if row[dd.branch_logic_index] == "" or row[dd.choice_index] == "":
        continue

      choices = translator.translate(row[dd.choice_index]).text.encode('utf-8')
      description = translator.translate(row[dd.description_index]).text.encode('utf-8')
      sepintlist = choices.split('|')
      choice_list = []
      for item in sepintlist:
        choice_list.append(int(re.search("(-?\d+),", item).group(1)))
      # Let the google translation server relax
      time.sleep(0.5)

      # Obtain current field specific information
      if row[dd.type_index] == "radio" or row[dd.type_index] == 'dropdown':
        feature_name = row[dd.feature_index].upper()
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

      elif row[dd.type_index] == 'checkbox':
        for c in choice_list:
          feature_name = row[dd.field_index].upper() + "__" + str(c)
          self.prefill_val_dict[feature_name] = 0


  # visitid is a string
  def filter(self, visitid, index):
    data = self.data_map[visitid][index]

    if data.no_ambiguous_data:
      data.df.replace(888, np.nan, inplace=True, regex=True)
      data.df.replace(999, np.nan, inplace=True, regex=True)
    print("Begin filtering...")
    print(data.file_name)
    print("The column number of raw data BEFORE filtered is: "+str(len(data.data_columns)))

    # Filter out data columns on feature basis
    removelist_column = []
    for column in data.data_columns:
      if column == study_id_feature:
        continue

      # Remove unchosen features
      if column not in self.data_dictionary.global_mixed_features:
        removelist_column.append(column)
        continue

      null_flags = data.df[column].isnull()
      valid_features_count = collections.Counter(null_flags)[False]
      valid_proportion_by_sample = float(valid_features_count) / float(len(null_flags))
      if valid_proportion_by_sample < self.completeness_threshold:
        removelist_column.append(column)
      else:
        # Find the specific field based on current data
        if column in self.data_dictionary.global_categorical_features:
          data.categorical_features.append(column)
        if column in self.data_dictionary.global_checkbox_features:
          data.checkbox_features.append(column)
        if column in self.data_dictionary.global_numerical_features:
          data.numerical_features.append(column)
    # Drop the columns
    data.df.drop(removelist_column, axis=1, inplace=True)
    # Update the data columns
    data.data_columns = data.df.columns.to_list()

    print("The column number of raw data AFTER filtered is: "+str(len(data.data_columns)))

    # Remove samples having too little valid features
    print("The row number of raw data BEFORE filtered is: "+str(len(data.data_indices)))
    removelist_idx = []
    for i in data.data_indices:
      data_row = data.df.loc[i]
      null_flags = data_row.isnull()
      valid_sample_count = collections.Counter(null_flags)[False]
      valid_proportion_by_feature = float(valid_sample_count) / float(len(null_flags))
      if valid_proportion_by_feature < self.completeness_threshold:
        removelist_idx.append(i)
    # Drop the rows
    data.df.drop(removelist_idx, axis=0, inplace=True)
    # Update the index
    data.df.reset_index(drop=True)
    data.data_indices = data.df.index.to_list()
    print("The row number of raw data AFTER filtered is: " + str(len(data.data_indices)))
    print("Filtering finished")



