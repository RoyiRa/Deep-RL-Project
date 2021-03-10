
class UserSettings:
    def __init__(self, batch_size, gamma, epsilon_decay, min_epsilon, is_soft_update, tau, episodes,
                 max_timestamp_per_episode, update_every_c_steps, reward_goal, min_number_of_episodes):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.is_soft_update = is_soft_update
        self.tau = tau
        self.episodes = episodes
        self.max_timestamp_per_episode = max_timestamp_per_episode
        self.update_every_c_steps = update_every_c_steps
        self.reward_goal = reward_goal
        self.min_number_of_episodes = min_number_of_episodes

