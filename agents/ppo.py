from networks.actor_critic import ActorCritic
from utils.device import get_device
from torch.distributions import Categorical
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
import random
from torch.nn.utils import clip_grad_norm_
import os

class PPO:
    def __init__(self, hidden_1, hidden_2, lr, epochs=10, update_frequency=2048, epsilon=0.2, l=0.95, gamma = 0.99, in_size=8, actor_out=4, value_out = 1):
        self.device = get_device()
        self.model = ActorCritic(hidden_1, hidden_2, in_size, actor_out, value_out).to(self.device)
        self.optimiser = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion_value = nn.MSELoss()
        self.experience_list = []
        self.step_count = 1
        self.update_frequency = update_frequency
        self.gamma = gamma
        self.epochs = epochs
        self.l = l
        self.epsilon = epsilon

    
    def act(self, obs):
        x = torch.from_numpy(obs).float().to(self.device)
        logits, _ = self.model(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(),log_prob

    def remember(self, obs, action, log_prob, reward, next_obs, done):
        obs = torch.tensor(obs).float().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        next_obs = torch.tensor(next_obs).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)
        self.experience_list.append((obs, action, log_prob, reward, next_obs, done))

    def reset(self):
        self.experience_list = []

    def update(self, done):
        if len(self.experience_list) == self.update_frequency:
            obs, action, log_prob, reward, next_obs, done = zip(*self.experience_list)
            obs = torch.stack(obs)
            action = torch.stack(action)
            next_obs = torch.stack(next_obs)
            reward = torch.stack(reward)
            done = torch.stack(done)
            log_prob = torch.stack(log_prob).detach()
            #GAE
            _, value = self.model(obs)
            value = value.detach()
            deltas = torch.zeros(len(self.experience_list)).to(self.device).detach()
            _, v_next = self.model(next_obs[-1])
            v_next = v_next.detach()
            for i in range(len(self.experience_list)-1,-1,-1):
                delta_t = reward[i] + self.gamma * v_next * (1-done[i]) - value[i]
                v_next = value[i]
                deltas[i] = delta_t
            advantage = torch.zeros(len(self.experience_list)).to(self.device).detach()
            A_next = advantage[-1]
            for i in range(len(self.experience_list)-1,-1,-1):
                A_t = deltas[i] + self.gamma * self.l * A_next * (1 - done[i])
                A_next = A_t
                advantage[i] = A_t
            advantage = (advantage - advantage.mean())/(advantage.std()+1e-8)
            target = advantage + value

                
            for epoch in range(self.epochs):
                self.optimiser.zero_grad()
                logits, value = self.model(obs)
                dist = Categorical(logits=logits)
                log_prob_new = dist.log_prob(action)
                entropy = dist.entropy()
                ratio = torch.exp(log_prob_new-log_prob)
                unclipped_obj = (ratio * advantage)
                clipped_ratio = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss =  - (torch.min(unclipped_obj, clipped_ratio))
                value_loss = self.criterion_value(target, value)
                loss = (0.5*value_loss + actor_loss - 0.01 * entropy).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimiser.step()
            print("Updated Parameters!")

            self.reset()


    def on_episode_end(self):
        pass

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def evaluate(self, obs, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        obs = torch.from_numpy(obs).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(obs)
            action = logits.argmax()
            action = logits.argmax().item()
            return action

    @property
    def name(self):
        return "PPO"
    
    