import random
from collections import deque

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from DDQN.utils.general_utils import *
from ddqn import quantize


class Agent:
    def __init__(self, env, optimizer, gamma):
        # general info

        self.state_size = env.observation_space.shape[0] # number of factors in the state; e.g: velocity, position, etc
        _, self.action_size = quantize(None)

        # allow large replay exp space
        self.replay_exp = deque(maxlen=1000000)

        self.gamma = gamma
        self.epsilon = 1.0  # initialize with high exploration, which will decay later

        # Build Policy Network
        self.brain_policy = Sequential()
        self.brain_policy.add(Dense(256, input_dim=self.state_size, activation="relu"))
        self.brain_policy.add(Dense(128, activation="relu"))
        self.brain_policy.add(Dense(self.action_size, activation="linear"))
        self.brain_policy.compile(loss="mse", optimizer=optimizer)

        # Build Target Network
        self.brain_target = Sequential()
        self.brain_target.add(Dense(256, input_dim=self.state_size, activation="relu"))
        self.brain_target.add(Dense(128, activation="relu"))
        self.brain_target.add(Dense(self.action_size, activation="linear"))
        self.brain_target.compile(loss="mse", optimizer=optimizer)

        self.update_brain_target()

    # add new experience to the replay exp
    def memorize_exp(self, state, action, reward, next_state, done):
        self.replay_exp.append((state, action, reward, next_state, done))

    def update_brain_target(self):
        return self.brain_target.set_weights(self.brain_policy.get_weights())

    def choose_action(self, state):
        if self.__should_do_exploration():
            action = np.random.choice(self.action_size)
        else:
            #TODO: ROY!!!! maybe should be self.state_size???
            state = np.reshape(state, [1, self.state_size])
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

        for index, exp in enumerate(mini_batch):
            sample_states[index] = exp[0]
            sample_actions[index] = exp[1]
            sample_rewards[index] = exp[2]
            sample_next_states[index] = exp[3]
            sample_dones[index] = exp[4]

        sample_qhat_next = self.brain_target.predict(sample_next_states)

        # set all Q values terminal states to 0
        sample_qhat_next = sample_qhat_next * (np.ones(shape=sample_dones.shape) - sample_dones)


        # choose max action for each state
        sample_qhat_next = np.max(sample_qhat_next, axis=1)

        sample_qhat = self.brain_policy.predict(sample_states)

        for i in range(cur_batch_size):
            a = tuple(sample_actions[i])
            sample_qhat[i, bucket_2_action[a]] = sample_rewards[i] + self.gamma * sample_qhat_next[i]

        q_target = sample_qhat
        self.brain_policy.fit(sample_states, q_target, epochs=1, verbose=0)


    def __should_do_exploration(self):
        return np.random.uniform(0.0, 1.0) < self.epsilon
