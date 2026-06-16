class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        action = self.action_space.sample()
        return action, None
    
    def remember(self, obs, action, log_prob, reward, next_obs, done):
        pass
    
    def update(self, done):
        pass

    def save(self, path):
        pass

    def evaluate(self, obs):
        action = self.action_space.sample()
        return action
    
    def on_episode_end(self):
        pass


    
