from configparser import ConfigParser


def load_user_config(path, section):
    config_object = ConfigParser()
    config_object.read(path)
    config_section = config_object[section]

    return config_section
