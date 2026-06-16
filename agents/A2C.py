from networks.actor_critic import ActorCritic
from utils.device import get_device
from torch.distributions import Categorical
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
import random

class A2C:
    def __init__(self, hidden_1, hidden_2, lr, update_frequency = 5, gamma=0.99, in_size=8, actor_out=4, value_out = 1):
        self.device = get_device()
        self.model = ActorCritic(hidden_1, hidden_2, in_size, actor_out, value_out).to(self.device)
        self.optimiser = optim.AdamW(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion_value = nn.MSELoss()
        self.experience_list = []
        self.step_count = 0
        self.update_frequency = update_frequency


    def act(self, obs):
        x = torch.from_numpy(obs).float().to(self.device)
        logits, _ = self.model(x)
        dist = Categorical(logits=logits) 
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def reset(self):
        self.experience_list = []
        

    def remember(self, obs, action, log_prob, reward, next_obs, done ):
        obs = torch.tensor(obs).float().to(self.device)
        next_obs = torch.tensor(next_obs).float().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)
        self.experience_list.append((obs, log_prob, reward, next_obs, done))

    def on_episode_end(self):
        pass

    def update(self, done):
        if self.step_count % self.update_frequency == 0:

            obs, log_prob, reward, next_obs, done = zip(*self.experience_list)
            self.optimiser.zero_grad()
            logits , value_obs = self.model(obs)
            dist = Categorical(logits=logits)
            log_probs = torch.log(dist.probs)
            _, value_next_obs = self.model(next_obs)
            target = reward + self.gamma*(value_next_obs) * (1-done)
            loss_value = self.criterion_value(target, value_obs)
            entropy = -(dist.probs * log_probs).sum(dim=-1)
            advantage = target - value_obs
            loss_actor = - log_prob * advantage
            loss = 0.5*loss_value + 0.5 * loss_actor + 0.1 * entropy
            loss.backward()
            self.optimiser.step()
            self.reset()


        self.step_count += 1
        




    def save(self):
        pass

    def evaluate(self):
        pass
