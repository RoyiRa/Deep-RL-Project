from collections import deque

import gym
import tensorflow.compat.v1 as tf

from DDQN.utils.general_utils import *
from DDQN.src.d3qn import D3QNAgent
from DDQN.src.ddqn import DDQNAgent

tf.disable_v2_behavior() # testing on tensorflow 1


def get_agent(env, optimizer, type='DDQN'):
    if type == 'D3QN':
        agent = D3QNAgent(env, optimizer)
    else:
        agent = DDQNAgent(env, optimizer)

    return agent


def main():
    env = gym.make(LANDER_CONTINUOUS)
    agent = get_agent(env, optimizer, type='DDQN')

    total_timesteps = 0
    rewards = []
    average_rewards = []
    aver = deque(maxlen=100)

    for episode in range(EPISODES):
        state = env.reset()
        state = add_noise(state)
        total_reward = 0
        timesteps = 0
        done = False
        while not done:
            if MAX_TIMESTEPS_PER_EPISODE <= timesteps:
                break

            action = agent.choose_action(state)
            action, _ = quantize(action)
            next_state, reward, done, info = env.step(action)  # returns sample
            next_state = add_noise(next_state)

            # reward = np.clip(reward, -1, 1)
            if agent.replay_exp.is_prioritized():
                agent.learn((state, action, reward, next_state, done))
                if BATCH_SIZE < agent.replay_exp.size:
                    agent.learn()
            else:
                agent.replay_exp.memorize_exp((state, action, reward, next_state, done), None)
                agent.learn()

            env.render()
            total_reward += reward
            state = next_state
            timesteps += 1
            total_timesteps += 1

            if timesteps % UPDATE_EVERY_C_STEPS == 0:
                agent.update_brain_target()

        aver.append(total_reward)
        average_rewards.append(np.mean(aver))
        rewards.append(total_reward)

        if BATCH_SIZE < agent.replay_exp.size:
            agent.epsilon = max(MIN_EPSILON, EPSILON_DECAY * agent.epsilon)  # decaying exploration
        print(episode, "\t: Episode || Reward: ", total_reward, "\t|| Average Reward: ", average_rewards[-1],"\t epsilon: ", agent.epsilon)
        print(f'Total timesteps: {total_timesteps}')

        if REWARD_GOAL <= average_rewards[-1] and MIN_NUMBER_OF_EPISODES <= len(average_rewards):
            print(f'Environment mastered in {episode} episodes!')
            break

    draw_plots(average_rewards, rewards)


if __name__ == '__main__':
    main()
