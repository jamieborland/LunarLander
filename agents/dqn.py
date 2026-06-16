from networks.MLP import MLP
from utils.device import get_device
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
import random
from collections import deque

class DQN:
    def __init__(self, hidden_1, hidden_2, lr, gamma, update_frequency, warm_up = 1000, buffer_size = 10000, batch_size = 64, in_size=8, out_size=4, epsilon=1.0, decay_rate=0.9999, min_eps = 0.01):
        self.device = get_device()
        self.onlineNet = MLP(hidden_1, hidden_2, in_size, out_size)
        self.onlineNet = self.onlineNet.to(self.device)
        self.targetNet = MLP(hidden_1, hidden_2, in_size, out_size)
        self.targetNet.load_state_dict(self.onlineNet.state_dict())
        self.targetNet = self.targetNet.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimiser = optim.AdamW(self.onlineNet.parameters(), lr=lr)
        self.gamma = gamma
        self.out_size = out_size
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_eps = min_eps
        self.batch_size = batch_size
        self.warm_up = warm_up
        self.replay_buffer = deque(maxlen=buffer_size)
        self.stepcount = 0
        self.update_frequency = update_frequency
        
        
    def act(self, obs):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.out_size)
        else:
            x = torch.from_numpy(obs).float().to(self.device)
            logits = self.onlineNet(x)
            action = torch.argmax(logits).item()
        return action, None
    
    def on_episode_end(self):
        self.epsilon = max(self.min_eps, self.epsilon*self.decay_rate)

    def remember(self, obs, action, log_prob, reward, next_obs, done):
        self.replay_buffer.append((obs, action, reward, next_obs, done))

    def update(self, done):
        if len(self.replay_buffer) < self.warm_up:
            return
        
        self.optimiser.zero_grad()
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs, actions, rewards, next_obs, done = zip(*batch)
        obs = torch.tensor(obs).float().to(self.device)
        next_obs = torch.tensor(next_obs).float().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        done = torch.tensor(done).float().to(self.device)
        next_actions = self.onlineNet(next_obs).argmax(dim=1)
        with torch.no_grad():
            targets = rewards.unsqueeze(1) + self.gamma*self.targetNet(next_obs).gather(1, next_actions.unsqueeze(1))*(1-done.unsqueeze(1))
        Q_s_a = self.onlineNet(obs).gather(1, actions.unsqueeze(1))
        loss = self.criterion(targets, Q_s_a)
        loss.backward()
        self.optimiser.step()

        if self.stepcount % self.update_frequency == 0:
            self.targetNet.load_state_dict(self.onlineNet.state_dict())
        self.stepcount += 1

    def save(self, path):
        torch.save({
            'online_net': self.onlineNet.state_dict(),
            'target_net': self.targetNet.state_dict()
        }, path)

    def evaluate(self, obs, path):
        checkpoint = torch.load(path, weights_only=True)
        self.onlineNet.load_state_dict(checkpoint['online_net'])
        self.targetNet.load_state_dict(checkpoint['target_net'])
        self.onlineNet.eval()
        self.targetNet.eval()
        x = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            logits = self.onlineNet(x)
            action = torch.argmax(logits).item()
        return action
        

    


