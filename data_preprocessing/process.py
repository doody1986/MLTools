from config.parser import *
from data_manager.manager import *
import missing_data_handler
import merger


# Read configuration
read_config()
print(config_.data_dict)
print(config_.data_path_map)
print(config_.label_file)

#Setup a data manager
prefill_path = '/home/shidong/Research/PROTECT/prefill_value_table.csv'
manager = Manager(config_.data_path_map,
                  config_.data_dict, config_.label_file, prefill_path)

# Update the label first
manager.update_label()

# Prefill for all
# manager.prefill_all()

# Filter for all
manager.filter_all()

# Internal merge for all
manager.merge_by_all_visit()

# Missing data handling
for visitid in manager.all_visits:
  manager.combined_data_by_visit[visitid].df = \
    missing_data_handler.handler(manager.combined_data_by_visit[visitid].df,
                                 manager.combined_data_by_visit[visitid].categorical_features,
                                 manager.combined_data_by_visit[visitid].checkbox_features,
                                 manager.combined_data_by_visit[visitid].numerical_features,
                                 manager.study_id_feature)

# Merge
manager.processed_data = merger.merger(manager.combined_data_by_visit,
                                       manager.study_id_feature,
                                       manager.all_visits)

if manager.processed_data is not None:
  manager.processed_data.to_csv("test.csv", index=False)


