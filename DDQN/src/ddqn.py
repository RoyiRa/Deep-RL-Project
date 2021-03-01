import gym
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from agent import *

tf.disable_v2_behavior() # testing on tensorflow 1

SETTINGS_SECTION = 'SETTINGS'
CONFIG_FILE_PATH = '../conf/config.ini'
LANDER_CONTINUOUS = 'LunarLanderContinuous-v2'


def quantize(signal):
    if signal is None:
        return None, len(all_discrete_actions)

    return all_discrete_actions[signal], len(all_discrete_actions)


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


def reward_clipping(reward):
    if reward < -1:
        reward = -1
    elif reward > 1:
        reward = 1
    return reward


def main():
    env = gym.make(LANDER_CONTINUOUS)
    settings = load_user_config(CONFIG_FILE_PATH, SETTINGS_SECTION)

    episodes, epsilon_decay, gamma, min_epsilon, min_number_of_episodes, reward_goal = read_configurations(settings)

    agent = Agent(env, optimizer, gamma)
    state_size = env.observation_space.shape[0]
    timestamp = 0
    rewards = []
    average_rewards = []
    aver = deque(maxlen=100)

    for episode in range(episodes):
        state = env.reset()

        noise = get_noise()
        state[0] += noise[0]
        state[1] += noise[1]

        total_reward = 0
        done = False
        while not done:

            action = agent.choose_action(state)
            action, _ = quantize(action)
            next_state, reward, done, info = env.step(action)  # returns sample
            noise = get_noise()
            next_state[0] += noise[0]
            next_state[1] += noise[1]

            reward = reward_clipping(reward)

            agent.memorize_exp(state, action, reward, next_state, done)

            # env.render()
            total_reward += reward
            agent.learn()

            state = next_state
            timestamp += 1

            # update model_target after each episode
            agent.update_brain_target()

        aver.append(total_reward)
        average_rewards.append(np.mean(aver))

        rewards.append(total_reward)

        if BATCH_SIZE < len(agent.replay_exp):
            agent.epsilon = max(min_epsilon, epsilon_decay * agent.epsilon)  # decaying exploration
        print(episode, "\t: Episode || Reward: ", total_reward, "\t|| Average Reward: ", average_rewards[-1],"\t epsilon: ", agent.epsilon)
        print(f'Total timestamp: {timestamp}')

        if reward_goal <= average_rewards[-1] and min_number_of_episodes <= len(average_rewards):
            print(f'Environment mastered in {episode} episodes!')
            break

    draw_plots(average_rewards, rewards)


if __name__ == '__main__':
    main()
