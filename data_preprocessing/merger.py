import collections

def merger(combined_data_by_visit, study_id_feature, visit_list):

  # Get the first data from visit_list
  assert(len(visit_list) > 1, "More than one visit data are needed!!!")
  assert(len(combined_data_by_visit) > 0,
         "The combined data for each visit cannot be empty! Internal merges are needed first!!!")
  visitid_left = visit_list[0]
  data = combined_data_by_visit[visitid_left]
  for idx in range(1, len(visit_list)):
    visitid_right = visit_list[idx]

    # Merge data
    data = data.merge(combined_data_by_visit[visitid_right],
                      on=study_id_feature, suffixes=(visitid_left, visitid_right))

  # Remove the STUDY_ID column
  data.drop(study_id_feature, axis=1, inplace=True)

  return data
