import numpy as np
import gym
import tensorflow.compat.v1 as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import random
from collections import deque
from utils import *

tf.disable_v2_behavior() # testing on tensorflow 1


def quantize(signal):
    if signal is None:
        return None, len(all_discrete_actions)

    return all_discrete_actions[signal], len(all_discrete_actions)


class Agent:
    def __init__(self, env, optimizer):
        # general info

        self.state_size = env.observation_space.shape[0] # number of factors in the state; e.g: velocity, position, etc
        _, self.action_size = quantize(None)
        self.optimizer = optimizer

        # allow large replay exp space
        self.replay_exp = deque(maxlen=1000000)

        self.gamma = 0.99
        self.epsilon = 1.0  # initialize with high exploration, which will decay later

        # Build Policy Network
        self.brain_policy = Sequential()
        self.brain_policy.add(Dense(256, input_dim=self.state_size, activation="relu"))
        self.brain_policy.add(Dense(128, activation="relu"))
        self.brain_policy.add(Dense(self.action_size, activation="linear"))
        self.brain_policy.compile(loss="mse", optimizer=self.optimizer)

        # Build Target Network
        self.brain_target = Sequential()
        self.brain_target.add(Dense(256, input_dim=self.state_size, activation="relu"))
        self.brain_target.add(Dense(128, activation="relu"))
        self.brain_target.add(Dense(self.action_size, activation="linear"))
        self.brain_target.compile(loss="mse", optimizer=self.optimizer)

        self.update_brain_target()

    # add new experience to the replay exp
    def memorize_exp(self, state, action, reward, next_state, done):
        self.replay_exp.append((state, action, reward, next_state, done))

    def update_brain_target(self):
        return self.brain_target.set_weights(self.brain_policy.get_weights())

    def choose_action(self, state):
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            state = np.reshape(state, [1, state_size])
            qhat = self.brain_policy.predict(state) # output Q(s,a) for all a of current state
            action = np.argmax(qhat[0])  # because the output is m * n, so we need to consider the dimension [0]

        return action

    def learn(self):
        # take a mini-batch from replay experience
        cur_batch_size = min(len(self.replay_exp), BATCH_SIZE)
        mini_batch = random.sample(self.replay_exp, cur_batch_size)

        # batch data
        sample_states = np.ndarray(shape=(cur_batch_size, self.state_size)) # replace 128 with cur_batch_size
        sample_actions = np.ndarray(shape=(cur_batch_size, 2))
        sample_rewards = np.ndarray(shape=(cur_batch_size, 1))
        sample_next_states = np.ndarray(shape=(cur_batch_size, self.state_size))
        sample_dones = np.ndarray(shape=(cur_batch_size, 1))

        temp = 0
        for exp in mini_batch:
            sample_states[temp] = exp[0]
            sample_actions[temp] = exp[1]
            sample_rewards[temp] = exp[2]
            sample_next_states[temp] = exp[3]
            sample_dones[temp] = exp[4]
            temp += 1

        sample_qhat_next = self.brain_target.predict(sample_next_states)


        # set all Q values terminal states to 0
        sample_qhat_next = sample_qhat_next * (np.ones(shape=sample_dones.shape) - sample_dones)


        # choose max action for each state
        sample_qhat_next = np.max(sample_qhat_next, axis=1)

        sample_qhat = self.brain_policy.predict(sample_states)

        errors = []
        for i in range(cur_batch_size):
            a = tuple(sample_actions[i])
            sample_qhat[i, bucket_2_action[a]] = sample_rewards[i] + self.gamma * sample_qhat_next[i]

        q_target = sample_qhat
        self.brain_policy.fit(sample_states, q_target, epochs=1, verbose=0)


env = gym.make("LunarLanderContinuous-v2")


agent = Agent(env, optimizer)
state_size = env.observation_space.shape[0]
timestep = 0
rewards = []
aver_reward = []
aver = deque(maxlen=100)

for episode in range(999999):
    state = env.reset()

    noise = get_noise()
    state[0] += noise[0]
    state[1] += noise[1]

    total_reward = 0
    done = False
    while not done:

        action = agent.choose_action(state)
        action, _ = quantize(action)
        next_state, reward, done, info = env.step(action) # returns sample
        noise = get_noise()
        next_state[0] += noise[0]
        next_state[1] += noise[1]
        if reward < -1:
            reward = -1
        elif reward > 1:
            reward = 1
        agent.memorize_exp(state, action, reward, next_state, done)




        env.render()
        total_reward += reward
        agent.learn()

        state = next_state
        timestep += 1

        # update model_target after each episode
        agent.update_brain_target()

    aver.append(total_reward)
    aver_reward.append(np.mean(aver))

    rewards.append(total_reward)

    if BATCH_SIZE < len(agent.replay_exp):
        agent.epsilon = max(0.1, 0.995 * agent.epsilon) # decaying exploration
    print(episode, "\t: Episode || Reward: ", total_reward, "\t|| Average Reward: ", aver_reward[-1],
          "\t epsilon: ", agent.epsilon)
    print(f'Total timesteps: {timestep}')

    if 200 <= aver_reward[-1] and 100 <= len(aver_reward):
        print(f'Environment mastered in {episode} episodes!')
        break

plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(rewards)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.plot(aver_reward, 'r')
