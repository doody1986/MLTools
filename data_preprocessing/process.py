from config.parser import *
from data_manager.manager import *
import missing_data_handler
import merger

# Read configuration
read_config()
print(config_.data_dict)
print(config_.data_path_map)
#print(config_.label_file)

# Setup a data manager
prefill_path = '/home/shidong/Research/PROTECT/prefill_value_table.csv'
manager = Manager(config_.data_path_map,
                  config_.data_dict, prefill_path)

# Prefill for all
manager.prefill_all()

# Filter for all
manager.filter_all()

# Internal merge for all
manager.merge_by_all_visit()

print(manager.combined_data_by_visit['V1'].df)

