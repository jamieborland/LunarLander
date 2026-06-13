from networks.reinforce_net import REINFORCE_NET
from utils.device import get_device
import torch.optim as optim
from torch.distributions import Categorical
import torch


class REINFORCE:
    def __init__(self, hidden_1, hidden_2, lr, gamma, in_size=8, out_size=4):
        self.device = get_device()
        self.model = REINFORCE_NET(hidden_1, hidden_2, in_size, out_size)
        self.model = self.model.to(self.device)
        self.optimiser = optim.AdamW(self.model.parameters(), lr=lr)
        self.rewards = []
        self.log_probs = []
        self.gamma = gamma

    def reset(self):
        self.log_probs = []
        self.rewards = []

    def act(self, obs):

        x = torch.from_numpy(obs).float()
        x = x.to(self.device)
        logits = self.model(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def remember(self, obs, action, log_prob,  reward, next_obs, done):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def update(self, done):
        if done:
            self.optimiser.zero_grad()
            returns = [0] * len(self.rewards)
            for i in range(len(self.rewards)-1, -1, -1):
                if i == len(self.rewards) - 1:
                    returns[i] = self.rewards[i]
                else:
                    returns[i] = self.rewards[i] + self.gamma*returns[i+1]
            returns = torch.tensor(returns).float()
            returns = returns.to(self.device)
            log_probs = torch.stack(self.log_probs)
            loss = -(log_probs*returns).sum()
            loss.backward()
            self.optimiser.step()
            self.reset()
