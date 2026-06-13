import numpy as np
import csv
import os

class Logger:
    def __init__(self, agent):
        self.history  = []
        self.agent = agent

    def log(self, episode, reward, loss = None):
        self.history.append({'episode': episode,
                         'reward' : reward,
                         'loss': loss})

    def average_reward(self, N = 100):
        rewards = [episode['reward'] for episode in self.history[-N:]]
        return np.mean(rewards)

    def save(self):
        path = f"results/{self.agent}/{self.agent}.csv"
        os.makedirs(f"results/{self.agent}", exist_ok=True)
        with open(path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'reward', 'loss'])
            writer.writeheader()
            writer.writerows(self.history)

        

