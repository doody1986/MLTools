import os
import yaml
from yaml import Loader

class Config:
  def __init__(self):
    self.data_dict = ""
    self.data_list = []
    self.label_file = ""

# A global configutration object
config_ = Config()

def read_config():
  # Prompt information
  prompt_info = \
  """ 
  The program is about the read data configurations based on a fixed data storing structure as below
  [Your current working directory]/
          data/
              data1.csv
              data2.csv
              ...
          data_dict/
              human_subjects_dd.csv
          label/ (optional)
              postpartum_data.csv
  Make sure all human subject data are stored in this way
  """
  print(prompt_info)

  # Fixed path for config file
  current_path = os.path.dirname(os.path.abspath(__file__))
  config_file_name = current_path+"/config.yaml"
  assert os.path.isfile(config_file_name), "Config file does not exist"

  # Load yaml config file
  cur_working_path = os.getcwd()
  with open(config_file_name, 'r') as f:
    config_yaml = yaml.load(f, Loader=Loader)

    # Obtain the DD file
    data_dict_dir = os.path.join(cur_working_path, config_yaml["data_dict_dir"])
    data_dict_name = os.path.join(data_dict_dir, config_yaml["data_dict_name"])
    assert os.path.isfile(data_dict_name), "Human subject data dictionary does not exist"
    config_.data_dict = data_dict_name

    # Obtain the data files
    data_dir = os.path.join(cur_working_path, config_yaml["data_dir"])
    assert os.path.isdir(data_dir), "Data directory does not exist"
    for fname in os.listdir(data_dir):
      data_file = os.path.join(data_dir, fname)
      if os.path.isdir(data_file):
        continue
      config_.data_list.append(data_file)
    assert len(config_.data_list) > 0, "No data found"

    # Obtain the label file
    label_dir = os.path.join(cur_working_path, config_yaml["label_file_dir"])
    label_name = os.path.join(label_dir, config_yaml["label_file_name"])
    if os.path.isfile(label_name):
      config_.label_file = label_name
    else:
      print("Warning: Label file does not exist, use the label in V4 data")
