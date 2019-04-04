from config.parser import *
from data_manager.manager import *
from missing_data_handler import *
from merger import *
import pandas as pd

# For test purpose only
read_config()
print(config_.data_dict)
print(config_.data_path_map)
#print(config_.label_file)

manager = Manager(config_.data_path_map, config_.data_dict)

