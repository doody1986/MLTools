from config.parser import *
from data_manager.manager import *

# Read configuration
read_config()
print(config_.data_dict)
print(config_.data_path_map)
print(config_.label_file)

completeness_threshold = 0.5

# Setup a data manager
manager = Manager(config_.data_path_map,
                  config_.data_dict, config_.label_file, "", completeness_threshold)

# Update the label first
manager.update_label()

# Filter for all
manager.filter_all()

# Generate entropy table
manager.calc_entropy()

entropy_table_filename = "entropy_table.csv"
manager.entropy_table.to_csv(entropy_table_filename, index=False)


