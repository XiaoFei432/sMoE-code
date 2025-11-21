import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticHead(nn.Module):
    def __init__(self, emb_dim, config_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim * 2 + config_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, g_emb, pooled_emb, action_emb):
        x = torch.cat([g_emb, pooled_emb, action_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.squeeze(-1)

class DoubleCritic(nn.Module):
    def __init__(self, emb_dim, config_dim, hidden_dim=128):
        super().__init__()
        self.q1 = CriticHead(emb_dim, config_dim, hidden_dim)
        self.q2 = CriticHead(emb_dim, config_dim, hidden_dim)

    def forward(self, g_emb, pooled_emb, action_emb):
        q1 = self.q1(g_emb, pooled_emb, action_emb)
        q2 = self.q2(g_emb, pooled_emb, action_emb)
        return q1, q2
