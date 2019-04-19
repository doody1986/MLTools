from config.parser import *
from data_manager.manager import *

# Read configuration
read_config()
print(config_.data_dict)
print(config_.data_path_map)
print(config_.label_file)

# Setup a data manager
prefill_path = '/home/shidong/Research/PROTECT/prefill_value_table.csv'
manager = Manager(config_.data_path_map,
                  config_.data_dict, config_.label_file, prefill_path)

# Generate missing rate table
manager.calc_missing_rate()

missing_rate_table_filename = "missing_rate_table.csv"
manager.missing_rate_table.to_csv(missing_rate_table_filename, index=False)


