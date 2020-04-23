import numpy as np
import re
import sys

regex_feature_suffix = re.compile(r".*(V\d)$")

def merger(combined_data_by_visit, study_id_feature, visit_list, label_updated,
           need_verify = False, datamap = None):
  print("====Merging Starts...====")
  # Get the first data from visit_list
  #assert(len(visit_list) > 1, "More than one visit data are needed!!!")
  #assert(len(combined_data_by_visit) > 0,
  #       "The combined data for each visit cannot be empty! Internal merges are needed first!!!")
  visitid_left = visit_list[0]
  data = combined_data_by_visit[visitid_left].df
  for idx in range(1, len(visit_list)):
    visitid_right = visit_list[idx]

    # Merge data
    data = data.merge(combined_data_by_visit[visitid_right].df,
                      on=study_id_feature, suffixes=(visitid_left, visitid_right))
    visitid_left = ''
  if 'V4' not in visit_list:
    if not label_updated:
      print("Warning: the labels are not updated!!!!!!!")
    # Attach labels from V4 data
    # Hardcode the label feature name PPTERM here
    label_df = combined_data_by_visit['V4'].df[[study_id_feature, 'PPTERM']].copy()
    data = data.merge(label_df, on=study_id_feature)

  # Verify the data if verify flag set to True
  num_verify = 200
  if need_verify:
    feature_list = data.columns.to_list()
    study_id_list = data[study_id_feature].to_list()
    if datamap is None:
      print("Error: The datamap has to be valid!!!!")
      exit()
    count = 0
    for i in np.random.randint(0, len(feature_list), num_verify):
      # Print the progress
      sys.stdout.write('\r>> Progress %.1f%%' % (float(count + 1) / num_verify * 100.0))
      sys.stdout.flush()
      current_feat = feature_list[i]
      implicit_visitid = ""
      if regex_feature_suffix.match(current_feat):
        current_feat_raw = current_feat[:-2]
        implicit_visitid = regex_feature_suffix.match(current_feat).group(1)
      else:
        current_feat_raw = current_feat
      for visitid in visit_list:
        for raw_data in datamap[visitid]:
          if current_feat_raw not in raw_data.data_columns:
            continue
          if implicit_visitid != visitid:
            continue
          current_df = raw_data.df
          for study_id in study_id_list:
            val_in_processed_data = data.loc[data[study_id_feature]==study_id, current_feat].values.tolist()[0]
            val_in_raw_data = current_df.loc[current_df[study_id_feature]==study_id, current_feat_raw].values.tolist()[0]
            if np.isnan(val_in_raw_data):
              continue
            if val_in_processed_data != val_in_raw_data:
              print("\nMissmatch found!!!!!")
              print("In "+visitid)
              print("Feature: "+current_feat)
              print("Study ID: "+str(study_id))
              print("Val in processed data: "+str(val_in_processed_data))
              print("Val in raw data: " + str(val_in_raw_data))
              exit()
      count += 1
    print("\nVerify Finished! PASS!")


  # Remove the STUDY_ID column
  data.drop(study_id_feature, axis=1, inplace=True)

  # Drop the row in which label is empty
  removelist_idx = []
  for i in data.index.to_list():
    if np.isnan(data.loc[i, 'PPTERM']):
      removelist_idx.append(i)
  data.drop(removelist_idx, axis=0, inplace=True)

  print("====Merging Finished...====\n")

  return data