import config

config.parser.read_config()
print(config.parser.config_.data_dict_name)