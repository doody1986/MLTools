
def merger(combined_data_by_visit, study_id_feature, visit_list, label_updated):
  print("====Merging Starts...====")
  # Get the first data from visit_list
  assert(len(visit_list) > 1, "More than one visit data are needed!!!")
  assert(len(combined_data_by_visit) > 0,
         "The combined data for each visit cannot be empty! Internal merges are needed first!!!")
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

  # Remove the STUDY_ID column
  data.drop(study_id_feature, axis=1, inplace=True)

  print("====Merging Finished...====\n")

  return data