from config.parser import *
from data_manager.manager import *
import missing_data_handler
import merger

completeness_threshold = 0.8

filename_prefix = "protect_data_no_prefill_"+str(int(completeness_threshold*100))

visit_options = [['V1', 'V2'], ['V1', 'V2', 'V3'], ['V1', 'V2', 'V3', 'V4']]

# Read configuration
read_config()
print(config_.data_dict)
print(config_.data_path_map)
print(config_.label_file)

#Setup a data manager
manager = Manager(config_.data_path_map,
                  config_.data_dict, config_.label_file,
                  "", completeness_threshold)

# Update the label first
manager.update_label()

# Filter for all
manager.filter_all()

# Internal merge for all
manager.merge_by_all_visit()

#Missing data handling
for visitid in manager.all_visits:
  print("In "+str(visitid))
  manager.combined_data_by_visit[visitid].df = \
    missing_data_handler.handler(manager.combined_data_by_visit[visitid].df,
                                 manager.combined_data_by_visit[visitid].categorical_features,
                                 manager.combined_data_by_visit[visitid].checkbox_features,
                                 manager.combined_data_by_visit[visitid].numerical_features,
                                 manager.study_id_feature)

# Merge all
# manager.processed_data = merger.merger(manager.combined_data_by_visit,
#                                        manager.study_id_feature,
#                                        manager.all_visits, manager.label_updated)

# Merge with verification enabled
# manager.processed_data = merger.merger(manager.combined_data_by_visit,
#                                        manager.study_id_feature,
#                                        ['V1', 'V2'], manager.label_updated,
#                                        True, manager.data_map)

# Merge based on the visit options
for opt in visit_options:
  filename = filename_prefix
  for visitid in opt:
    filename += '_'+visitid
  manager.processed_data = merger.merger(manager.combined_data_by_visit,
                                         manager.study_id_feature,
                                         opt, manager.label_updated)

  if manager.processed_data is not None:
    manager.processed_data.to_csv(filename+".csv", index=False)
  manager.processed_data = None


