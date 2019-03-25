import config

config.parser.read_config()
print(config.parser.config_.data_dict)
print(config.parser.config_.data_list)
print(config.parser.config_.label_file)