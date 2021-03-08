import itertools
import matplotlib.pyplot as plt

import numpy as np

# Quantization vars:
M_ENGINE = [-1., 0.2, 0.6]
S_ENGINE = [-0.57, 0., 0.57]

DISCRETE_ACTIONS = list(itertools.product(*[M_ENGINE, S_ENGINE]))

ACTIONS_TO_IDX = {bucket: idx for idx, bucket in enumerate(DISCRETE_ACTIONS)}


def quantize(signal):
    if signal is None:
        return None, len(DISCRETE_ACTIONS)

    return DISCRETE_ACTIONS[signal], len(DISCRETE_ACTIONS)


def add_noise(state):
    noise = np.random.normal(0, 0.05, 2)
    state[0] += noise[0]
    state[1] += noise[1]
    return state


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

