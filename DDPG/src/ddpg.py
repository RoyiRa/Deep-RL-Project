import gym
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

from DDPG.src.noise import OUNoise
from DDPG.src.actor import ActorNetwork
from DDPG.src.critic import CriticNetwork
from DDPG.src.replay_buffer import ReplayBuffer
from DDPG.utils.general_utils import load_user_config

EXPECTED_REWARD = 200

MAX_STEPS = 3000

SETTINGS_SECTION = 'SETTINGS'
CONFIG_FILE_PATH = '../conf/config.ini'

LANDER_CONTINUOUS = 'LunarLanderContinuous-v2'


def train(sess, environment, actor, critic, actor_noise, buffer_size, min_batch, episodes):
    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(buffer_size, 0)

    max_episodes = episodes
    max_steps = MAX_STEPS
    score_list = []

    for i in range(max_episodes):

        state = environment.reset()
        score = 0

        for j in range(max_steps):
            # environment.render()

            action = actor.predict(np.reshape(state, (1, actor.s_dim))) + actor_noise()
            next_state, reward, done, info = environment.step(action[0])
            replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)),
                              reward, done, np.reshape(next_state, (actor.s_dim,)))

            # updating the network in batch
            if replay_buffer.size() < min_batch:
                continue

            states, actions, rewards, dones, next_states = replay_buffer.sample_batch(min_batch)
            target_q = critic.predict_target(next_states, actor.predict_target(next_states))

            y = []
            for k in range(min_batch):
                y.append(rewards[k] + critic.gamma * target_q[k] * (1-dones[k]))

            # Update the critic given the targets
            predicted_q_value, _ = critic.train(states, actions, np.reshape(y, (min_batch, 1)))

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(states)
            grads = critic.action_gradients(states, a_outs)
            actor.train(states, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

            state = next_state
            score += reward

            if done:
                print('Reward: {} | Episode: {}/{}'.format(int(score), i, max_episodes))
                break

        score_list.append(score)

        avg = np.mean(score_list[-100:])
        print("Average of last 100 episodes: {avg} \n".format(avg="{0:.2f}".format(avg)))

        if avg > EXPECTED_REWARD:
            print(f'Task Completed with average reward: {avg}')
            break

    return score_list


def init_environment():
    env = gym.make(LANDER_CONTINUOUS)
    env.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    return env


def read_config_data():
    settings = load_user_config(CONFIG_FILE_PATH, SETTINGS_SECTION)

    episodes = int(settings["episodes"])
    tau = float(settings["tau"])
    gamma = float(settings["gamma"])
    min_batch = int(settings["min_batch"])
    actor_lr = float(settings["actor_lr"])
    critic_lr = float(settings["critic_lr"])
    buffer_size = int(settings["buffer_size"])

    return actor_lr, buffer_size, critic_lr, episodes, min_batch, tau, gamma


def plot_scores_graph(scores):
    plt.plot([i + 1 for i in range(0, len(scores), 4)], scores[::4])
    plt.show()


def main():
    with tf.Session() as sess:
        environment = init_environment()

        actor_lr, buffer_size, critic_lr, episodes, min_batch, tau, gamma = read_config_data()

        state_dim = environment.observation_space.shape[0]
        action_dim = environment.action_space.shape[0]
        action_bound = environment.action_space.high

        actor_noise = OUNoise(mu=np.zeros(action_dim))
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, actor_lr, tau, min_batch)
        critic = CriticNetwork(sess, state_dim, action_dim, critic_lr, tau, gamma, actor.get_num_trainable_vars())
        scores = train(sess, environment, actor, critic, actor_noise, buffer_size, min_batch, episodes)

        plot_scores_graph(scores)


if __name__ == '__main__':
    main()

