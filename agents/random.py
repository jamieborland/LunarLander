class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()
    
    def remember(self, obs, action, reward, next_obs, done):
        pass
    
    def update(self,):
        pass

    
