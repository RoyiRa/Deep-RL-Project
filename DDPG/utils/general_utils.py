import itertools
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam


def quantize(signal):
    if signal is None:
        return None, len(DISCRETE_ACTIONS)

    return DISCRETE_ACTIONS[signal], len(DISCRETE_ACTIONS)


def add_noise(state):
    noise = np.random.normal(0, 0.05, 2)
    state[0] += noise[0]
    state[1] += noise[1]
    return state


def load_user_config(path, section):
    config_object = ConfigParser()
    config_object.read(path)
    config_section = config_object[section]

    return config_section

def draw_plots(aver_reward, rewards):
    plt.title("Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rewards)
    plt.show()

    plt.title("Average Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.plot(aver_reward, 'r')
    plt.show()


def read_configurations(settings):
    epsilon_decay = float(settings["epsilon_decay"])
    min_epsilon = float(settings["min_epsilon"])
    episodes = int(settings["episodes"])
    reward_goal = int(settings["reward_goal"])
    min_number_of_episodes = int(settings["min_number_of_episodes"])
    gamma = float(settings["gamma"])
    return episodes, epsilon_decay, gamma, min_epsilon, min_number_of_episodes, reward_goal


# General Vars
LANDER_CONTINUOUS = 'LunarLanderContinuous-v2'
SETTINGS_SECTION = 'SETTINGS'
CONFIG_FILE_PATH = '../conf/config.ini'

MAX_TIMESTEPS_PER_EPISODE = 2000
UPDATE_EVERY_C_STEPS = 100
EPISODES = 99999
REWARD_GOAL = 200
MIN_NUMBER_OF_EPISODES = 100

# Replay vars
REPLAY_REGULAR = 'regular'
REPLAY_PRIORITIZED = 'prioritized'
REPLAY_TYPE = REPLAY_REGULAR

# NN vars
optimizer = Adam(learning_rate=0.0001)
BATCH_SIZE = 64
BUFFER_SIZE = int(1e6)
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
IS_SOFT_UPDATE = False
TAU = 0.001 # target network soft update hyperparameter

# Quantization vars
m_engine = [-1., 0.2, 0.6]
s_engine = [-0.57, 0., 0.57]
DISCRETE_ACTIONS = list(itertools.product(*[m_engine, s_engine]))
actions_2_idx = {bucket: idx for idx, bucket in enumerate(DISCRETE_ACTIONS)}
