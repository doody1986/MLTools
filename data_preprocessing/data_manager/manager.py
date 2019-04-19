import collections
import re
import pandas as pd
import numpy as np
import copy
import sys
import os


logic_regex_str = r"\[(\w+)\]\s?(=|<>|<|>|>=|<=)\s?'(\d+)'\s?(or|and)?"
regex_checkbox_feature = re.compile(r".*(__)\d+")

# Universal feature names
useless_features = ['FACILITY_ID', 'SUBFACILITY_CODE', 'REDCAP_EVENT_NAME', 'EBATCH']

# Form to visit map
form_to_visit_dict = {'first_visit':'V1', 'med_rec_v1':'V1', 'inhome_visit':'V2',
                      'inhome_visit_2nd_part':'V2', '':'V'}



class Data:
  def __init__(self, data_file, no_ambiguous_data = False):
    self.file_name = data_file
    self.df = pd.read_csv(data_file, low_memory=False)

    self.data_columns = self.df.columns.to_list()
    self.data_indices = self.df.index.to_list()

    # Directly drop the useless features from SQL server
    real_useless_features = [x for x in useless_features if x in self.data_columns]
    for s in self.data_columns:
      if 'COMPLETE' in s:
        real_useless_features.append(s)
    self.df.drop(real_useless_features, axis=1, inplace=True)

    # Update the data columns
    self.data_columns = self.df.columns.to_list()

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

class Label:
  def __init__(self, label_file):
    self.df = pd.read_csv(label_file)
    # Very file specific, not general
    self.df = self.df.replace('NA', np.nan, regex=True)
    self.study_id_feature = 'studyid'
    self.label_feature = 'preterm_best'


class Manager:
  def __init__(self, data_path_map, data_dictionary_path, label_path, prefill_table_path=''):
    # Extract data dictionary
    self.data_dictionary = DataDict(data_dictionary_path)
    self.extract_dd()

    # Extract label file
    if label_path != "":
      self.labels = Label(label_path)
    else:
      self.labels = None
    
    self.data_map = {}
    self.combined_data_by_visit = {}

    # Get data
    for visitid in data_path_map:
      if visitid not in self.data_map:
        self.data_map[visitid] = []
      for data_file in data_path_map[visitid]:
        data = Data(data_file)
        self.data_map[visitid].append(data)
      if visitid not in self.combined_data_by_visit:
        self.combined_data_by_visit[visitid] = None

    self.completeness_threshold = 0.8
    self.prefill_table_path = prefill_table_path
    self.prefill_val_df = pd.read_csv(self.prefill_table_path)
    self.study_id_feature = 'SYS_LOC_CODE'
    self.all_visits = ['V1', 'V2', 'V3', 'V4']
    self.label_feature = 'PPTERM'
    self.label_visit = 'V4'
    self.processed_data = None

    self.label_updated = False

    self.missing_rate_table = pd.DataFrame(columns=['Features', 'Missing Rate', 'Visit'])

  def extract_dd(self):
    print("====Data Dictionary Extraction Starts...====")
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
          found_int = re.search("-?\d+", item)
          dd.global_checkbox_features.append(row[dd.feature_index].upper() + "__" + str(found_int.group()))
          dd.global_mixed_features.append(row[dd.feature_index].upper() + "__" + str(found_int.group()))
      else:
        dd.global_text_features.append(row[dd.feature_index].upper())
    print("====Data Dictionary Extraction Finished====\n")

  def generate_prefill_val_table_for_features_with_deps(self):
    # Need further action to make use of files
    print("====Generate Prefill Value Table Starts...====")
    dd = self.data_dictionary

    # Features in the prefill table
    headers = self.prefill_val_df.columns.to_list()
    features_prefill_table = self.prefill_val_df[headers[0]].to_list()

    # Human subject data
    num_idx = len(dd.dd_indices)
    for idx in dd.dd_indices:
      # Print the progress
      sys.stdout.write('\r>> Progress %.1f%%' % (float(idx + 1) / float(num_idx) * 100.0))
      sys.stdout.flush()

      row = dd.df.loc[idx].tolist()
      depending_logic = row[dd.branch_logic_index]
      choices = row[dd.choice_index]
      if depending_logic == "" or choices == "":
        continue

      # Skip this feature if we already have it
      feature_name = row[dd.feature_index].upper()
      if feature_name in features_prefill_table:
        continue

      # Special case for checkbox feature
      checkbox_features = [m.group(0) for l in features_prefill_table
                           for m in [regex_checkbox_feature.search(l)] if m]
      skip = False
      for feature in checkbox_features:
        if feature_name in feature:
          skip = True
      if skip:
        continue

      sepintlist = choices.split('|')
      choice_list = []
      for item in sepintlist:
        choice_list.append(int(re.search("(-?\d+),", item).group(1)))

      # Obtain current field specific information
      prefill_val = 0
      if row[dd.type_index] == "radio" or row[dd.type_index] == 'dropdown':
        if 0 not in choice_list:
          prefill_val = 0
        elif 999 in choice_list:
          prefill_val = 999
        elif 999 not in choice_list and 888 not in choice_list:
          prefill_val = min(choice_list) - 1
        elif min(choice_list) < 0:
          prefill_val = 0
        self.prefill_val_df = \
          self.prefill_val_df.append({headers[0]: feature_name,
                                      headers[1]: prefill_val,
                                      headers[2]: depending_logic}, ignore_index=True)

      elif row[dd.type_index] == 'checkbox':
        for c in choice_list:
          feature_name = row[dd.feature_index].upper() + "__" + str(c)
          self.prefill_val_df = \
            self.prefill_val_df.append({headers[0]: feature_name,
                                        headers[1]: 0,
                                        headers[2]: depending_logic}, ignore_index=True)
    if self.prefill_table_path:
      self.prefill_val_df.to_csv(self.prefill_table_path, index=False)
    else:
      cur_working_path = os.getcwd()
      self.prefill_table_path = os.path.join(cur_working_path, "prefill_value_table.csv")
      self.prefill_val_df.to_csv(self.prefill_table_path, index=False)
    print("\n====Generate Prefill Value Table Finished====\n")

  def prefill_all(self):
    for visitid in self.all_visits:
      for i in range(len(self.data_map[visitid])):
        self.prefill(visitid, i)

  def prefill(self, visitid, index):
    # Need further action to make use of files
    print("====Prefill Starts...====")
    print("Visit: " + visitid)

    dd = self.data_dictionary
    prefill_table = self.prefill_val_df
    data = self.data_map[visitid][index]
    print("Form: " + data.file_name)

    # Header - 0: Features 1: Pre-Filled Value 2: Depending Logic
    headers = self.prefill_val_df.columns.to_list()
    feature_set = prefill_table[headers[0]].to_list()

    # Human subject data
    num_idx = len(data.data_indices)
    for idx in data.data_indices:
      # Print the progress
      sys.stdout.write('\r>> Progress %.1f%%' % (float(idx + 1) / float(num_idx) * 100.0))
      sys.stdout.flush()
      for feature in data.data_columns:
        if feature not in feature_set:
          continue
        if not pd.isnull(data.df.loc[idx, feature]):
          continue
        depending_logic = self.prefill_val_df.loc[self.prefill_val_df[headers[0]]==feature,
                                              headers[2]].to_list()[0]
        prefill_val = self.prefill_val_df.loc[self.prefill_val_df[headers[0]]==feature,
                                              headers[1]].to_list()[0]
        matches = re.findall(logic_regex_str, depending_logic)
        logic_operator = ''
        # Two conditions
        if len(matches) == 1:
          condition_feat = matches[0][0].upper()
          condition_comp = matches[0][1]
          condition_val = matches[0][2]
          if condition_comp == "=" and data.df.loc[idx, condition_feat] != int(condition_val):
            data.df.loc[idx, feature] = prefill_val
          if condition_comp == "<>" and data.df.loc[idx, condition_feat] == int(condition_val):
            data.df.loc[idx, feature] = prefill_val
          if condition_comp == ">=" and data.df.loc[idx, condition_feat] < int(condition_val):
            data.df.loc[idx, feature] = prefill_val
          assert(not pd.isnull(data.df.loc[idx, feature]), "Still NAN")
        elif len(matches) == 2:
          # Ignore so far TBD
          continue
          #logic_operator = matches[0][3]
          #assert(len(matches)==4, "No logic operator")
          #condition1_feat = matches[0][0]
          #condition1_comp = matches[0][1]
          #condition1_val = matches[0][2]
          #condition2_feat = matches[1][0]
          #condition2_comp = matches[1][1]
          #condition2_val = matches[1][2]

    print("\n====Prefill Finished====\n")

  def filter_all(self):
    for visitid in self.all_visits:
      for i in range(len(self.data_map[visitid])):
        self.filter(visitid, i)

  # visitid is a string
  def filter(self, visitid, index):
    print("====Data Filtering Starts...====")
    print("Visit: "+visitid)

    data = self.data_map[visitid][index]
    print("Form: " +data.file_name)

    if data.no_ambiguous_data:
      data.df.replace(888, np.nan, inplace=True, regex=True)
      data.df.replace(999, np.nan, inplace=True, regex=True)
    print(data.file_name)
    print("The column number of raw data BEFORE filtered is: "+str(len(data.data_columns)))

    # Filter out data columns on feature basis
    removelist_column = []
    for column in data.data_columns:
      if column == self.study_id_feature:
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
    print("====Data Filtering Finished====\n")

  def merge_by_all_visit(self):
    for visitid in self.all_visits:
      self.merge_by_visit(visitid)

  def merge_by_visit(self, visitid):
    print("====Internal Merge Starts...====")
    print("Visit: "+visitid)
    data_per_visit = self.data_map[visitid]
    num_data = len(data_per_visit)

    # Use copy to avoid modify original data
    local_data  = copy.deepcopy(data_per_visit[0])

    for i in range(1, num_data):
      local_data.df = local_data.df.merge(data_per_visit[i].df, on=self.study_id_feature)
      local_data.categorical_features = local_data.categorical_features + data_per_visit[i].categorical_features
      local_data.checkbox_features = local_data.checkbox_features + data_per_visit[i].checkbox_features
      local_data.numerical_features = local_data.numerical_features + data_per_visit[i].numerical_features
    local_data.data_columns = local_data.df.columns.to_list()
    local_data.data_indices = local_data.df.index.to_list()
    self.combined_data_by_visit[visitid] = local_data
    print("The number of rows: "+str(len(self.combined_data_by_visit[visitid].data_indices)))
    print("The number of columns: " + str(len(self.combined_data_by_visit[visitid].data_columns)))
    print("====Internal Merge Finished...====\n")


  def update_label(self):
    print("====Update label Starts...====")
    if self.labels is None:
      print("====No label files, update aborted====\n")
      return
    if self.label_visit not in self.data_map:
      print("====No V4 data forms, update aborted====\n")
      return
    old_labels = self.data_map[self.label_visit][0]
    old_labels_df = old_labels.df[[self.study_id_feature,
                                   self.label_feature]]
    new_labels = self.labels
    new_labels_df = new_labels.df[[self.labels.study_id_feature,
                                   self.labels.label_feature]]
    new_labels_indices = new_labels_df.index.to_list()
    for idx in new_labels_indices:
      # Print the progress
      sys.stdout.write('\r>> Progress %.1f%%' % (float(idx + 1) / float(len(new_labels_indices)) * 100.0))
      sys.stdout.flush()
      label_new = new_labels_df.loc[idx, new_labels.label_feature] + 1.0
      study_id = new_labels_df.loc[idx, new_labels.study_id_feature]
      if len(old_labels_df[old_labels_df[self.study_id_feature] == study_id].index.to_list()) > 0:
        idx_in_old_labels = old_labels_df[old_labels_df[self.study_id_feature] == study_id].index.to_list()[0]
      label_old = old_labels_df.loc[idx_in_old_labels, self.label_feature]

      #print label_new
      if label_new == label_old or np.isnan(label_new):
        continue
      else:
        old_labels.df.loc[idx_in_old_labels, self.label_feature] = label_new
    self.label_updated = True
    print("\n====Update label Finished====\n")

  def calc_missing_rate(self):
    print("====Calculate Missing Rate...====")
    for visitid in self.data_map:
      for data in self.data_map[visitid]:
        for feat in data.data_columns:
          null_flags = data.df[feat].isnull()
          invalid_count = collections.Counter(null_flags)[True]
          missing_rate = float(invalid_count) / float(len(null_flags))
          temp = pd.DataFrame([[feat, missing_rate, visitid]], columns=self.missing_rate_table.columns)
          self.missing_rate_table = self.missing_rate_table.append(temp, ignore_index=True)
    print("\n====Calculate Missing Rate Finished====\n")

