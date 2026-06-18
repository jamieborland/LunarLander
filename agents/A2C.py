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

class A2C:
    def __init__(self, hidden_1, hidden_2, lr, n_step=False, update_frequency = 5, gamma=0.99, in_size=8, actor_out=4, value_out = 1):
        self.device = get_device()
        self.model = ActorCritic(hidden_1, hidden_2, in_size, actor_out, value_out).to(self.device)
        self.optimiser = optim.AdamW(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion_value = nn.MSELoss()
        self.experience_list = []
        self.step_count = 1
        self.update_frequency = update_frequency
        self.n_step = n_step



    def act(self, obs):
        x = torch.from_numpy(obs).float().to(self.device)
        logits, _ = self.model(x)
        dist = Categorical(logits=logits) 
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob
    
    def reset(self):
        self.experience_list = []
        

    def remember(self, obs, action, log_prob, reward, next_obs, done ):
        obs = torch.tensor(obs).float().to(self.device)
        next_obs = torch.tensor(next_obs).float().to(self.device)
        reward = torch.tensor(reward).float().to(self.device).clamp(-1, 1)
        done = torch.tensor(done).float().to(self.device)
        self.experience_list.append((obs, log_prob, reward, next_obs, done))

    def on_episode_end(self):
        self.experience_list = []

    def update(self, done):
        if (self.step_count % self.update_frequency == 0 or done) and len(self.experience_list) > 0:

                obs, log_prob, reward, next_obs, done = zip(*self.experience_list)
                obs = torch.stack(obs)
                next_obs = torch.stack(next_obs)
                reward = torch.stack(reward)
                done = torch.stack(done)
                log_prob = torch.stack(log_prob)
                self.optimiser.zero_grad()
                logits , value_obs = self.model(obs)
                dist = Categorical(logits=logits)
                entropy = dist.entropy()

                if self.n_step:
                    target = torch.zeros(len(self.experience_list)).to(self.device)
                    _, value_last_obs = self.model(next_obs[-1])
                    V_next = value_last_obs.detach()
                    for i in range(len(self.experience_list) - 1, -1, -1):
                        y = reward[i]+ self.gamma*V_next * (1-done[i])
                        V_next = y
                        target[i] = y

                else:
                    _, value_next_obs = self.model(next_obs)
                    target = (reward + self.gamma*(value_next_obs) * (1-done)).detach()

                loss_value = self.criterion_value(target, value_obs)
                advantage = target - value_obs.detach()
                advantage = (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8)
                loss_actor = - log_prob * advantage
                loss = (0.5*loss_value + loss_actor - 0.01 * entropy).mean()
                #print(f"V(s): {value_obs.mean().item():.1f}  target: {target.mean().item():.1f}  adv_raw: {(target - value_obs.detach()).mean().item():.1f}  loss_v: {loss_value.item():.3f}  loss_pi: {loss_actor.mean().item():.3f}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)           
                self.optimiser.step()
                self.reset()

        self.step_count += 1
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def evaluate(self, obs, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        obs = torch.from_numpy(obs).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(obs)
            action = logits.argmax().item()
            return action
        
    @property
    def name(self):
        return "A2C_nstep" if self.n_step else "A2C_td1"
            
            
