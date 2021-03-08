import itertools
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy as np


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
