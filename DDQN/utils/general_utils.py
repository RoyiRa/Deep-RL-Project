from configparser import ConfigParser

from tensorflow.keras.optimizers import Adam

SETTINGS_SECTION = 'SETTINGS'
CONFIG_FILE_PATH = '../conf/config.ini'


def load_user_config(path, section):
    config_object = ConfigParser()
    config_object.read(path)
    config_section = config_object[section]

    return config_section


def read_configurations():
    settings = load_user_config(CONFIG_FILE_PATH, SETTINGS_SECTION)

    batch_size = int(settings["BATCH_SIZE"])
    gamma = float(settings["GAMMA"])
    epsilon_decay = float(settings["EPSILON_DECAY"])
    min_epsilon = float(settings["MIN_EPSILON"])
    is_soft_update = bool(settings["IS_SOFT_UPDATE"])
    tau = float(settings["TAU"])
    episodes = int(settings["EPISODES"])
    max_timestamp_per_episode = float(settings["MAX_TIMESTAMPS_PER_EPISODE"])
    update_every_c_steps = float(settings["UPDATE_EVERY_C_STEPS"])
    reward_goal = float(settings["REWARD_GOAL"])
    min_number_of_episodes = int(settings["MIN_NUMBER_OF_EPISODES"])

    return batch_size, gamma, epsilon_decay, min_epsilon, is_soft_update, tau, episodes,\
           max_timestamp_per_episode, update_every_c_steps, reward_goal, min_number_of_episodes


# General Vars
LANDER_CONTINUOUS = 'LunarLanderContinuous-v2'

BUFFER_SIZE = 1000000


# Replay vars
REPLAY_PRIORITIZED = 'prioritized'
REPLAY_REGULAR = 'regular'
REPLAY_TYPE = REPLAY_REGULAR

# NN vars
optimizer = Adam(learning_rate=0.0001)

