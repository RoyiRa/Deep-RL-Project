import gym
import tensorflow.compat.v1 as tf
from collections import deque
from DDQN.utils.general_utils import *
from DDQN.utils.ddqn_utils import *
from DDQN.src.d3qn import D3QNAgent
from DDQN.src.ddqn import DDQNAgent

tf.disable_v2_behavior() # testing on tensorflow 1


def get_agent(env, optimizer,gamma,batch_size, type='DDQN'):
    if type == 'D3QN':
        agent = D3QNAgent(env, optimizer, gamma)
    else:
        agent = DDQNAgent(env, optimizer, gamma, batch_size)

    return agent


def main():
    env = gym.make(LANDER_CONTINUOUS)

    batch_size, gamma, epsilon_decay, min_epsilon, is_soft_update, tau, episodes, \
    max_timestamp_per_episode, update_every_c_steps, reward_goal, min_number_of_episodes = read_configurations()

    agent = get_agent(env, optimizer, gamma, batch_size, type='DDQN')

    total_timestamps = 0
    rewards = []
    average_rewards = []
    aver = deque(maxlen=100)

    for episode in range(episodes):
        state = env.reset()
        state = add_noise(state)
        total_reward = 0
        timesteps = 0
        done = False
        while not done:
            if max_timestamp_per_episode <= timesteps:
                break

            action = agent.choose_action(state)
            action, _ = quantize(action)
            next_state, reward, done, info = env.step(action)  # returns sample
            next_state = add_noise(next_state)

            # reward = np.clip(reward, -1, 1)
            if agent.replay_exp.is_prioritized():
                agent.learn((state, action, reward, next_state, done))
                if batch_size < agent.replay_exp.size:
                    agent.learn()
            else:
                agent.replay_exp.memorize_exp((state, action, reward, next_state, done), None)
                agent.learn()

            env.render()
            total_reward += reward
            state = next_state
            timesteps += 1
            total_timestamps += 1

            if timesteps % update_every_c_steps == 0:
                agent.update_brain_target()

        aver.append(total_reward)
        average_rewards.append(np.mean(aver))
        rewards.append(total_reward)

        if batch_size < agent.replay_exp.size:
            agent.epsilon = max(min_epsilon, epsilon_decay * agent.epsilon)  # decaying exploration
        print(episode, "\t: Episode || Reward: ", total_reward, "\t|| Average Reward: ", average_rewards[-1],"\t epsilon: ", agent.epsilon)
        print(f'Total timestamps: {total_timestamps}')

        if reward_goal <= average_rewards[-1] and min_number_of_episodes <= len(average_rewards):
            print(f'Environment mastered in {episode} episodes!')
            break

    draw_plots(average_rewards, rewards)


if __name__ == '__main__':
    main()
