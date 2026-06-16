import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, hidden_1, hidden_2, in_size=8, actor_out=4, value_out = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.actor_out = nn.Linear(hidden_2, actor_out)
        self.value_out = nn.Linear(hidden_2, value_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor_out(x)
        value = self.value_out(x).squeeze(-1)

        return logits, value


    