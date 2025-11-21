import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from smoe.models.dagnn import SimpleDAGNN
from smoe.models.pointer_actor import PointerActor
from smoe.models.critic import DoubleCritic
from smoe.utils.graph_utils import split_expert_embeddings_by_layer

Transition = collections.namedtuple("Transition", ["obs", "action_emb", "reward", "next_obs", "done"])

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class GPRLAgent:
    def __init__(self, graph, obs_dim, config_dim, dagnn_hidden=64, dagnn_layers=3, pointer_hidden=64, siso_hidden=128, actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005, behavior_reg=0.01, device="cpu"):
        self.g = graph
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.behavior_reg = behavior_reg
        self.dagnn = SimpleDAGNN(obs_dim, dagnn_hidden, dagnn_layers).to(device)
        self.actor = PointerActor(dagnn_hidden, config_dim, pointer_hidden, siso_hidden).to(device)
        self.critic = DoubleCritic(dagnn_hidden, config_dim).to(device)
        self.target_dagnn = SimpleDAGNN(obs_dim, dagnn_hidden, dagnn_layers).to(device)
        self.target_actor = PointerActor(dagnn_hidden, config_dim, pointer_hidden, siso_hidden).to(device)
        self.target_critic = DoubleCritic(dagnn_hidden, config_dim).to(device)
        self.target_dagnn.load_state_dict(self.dagnn.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay = ReplayBuffer()

    def select_actions(self, obs, layer_config_tensors, epsilon=0.0):
        self.dagnn.eval()
        self.actor.eval()
        with torch.no_grad():
            node_embs, g_emb = self.dagnn(self.g, obs)
            layer_embs = split_expert_embeddings_by_layer(self.g, node_embs)
            actions, _ = self.actor(layer_embs, g_emb, layer_config_tensors)
        if epsilon > 0.0:
            for l in actions:
                for i in range(len(actions[l])):
                    if random.random() < epsilon:
                        m = layer_config_tensors[l].size(1)
                        actions[l][i] = random.randrange(m)
        return actions

    def _build_action_embedding(self, layer_config_tensors, actions):
        vecs = []
        for layer, confs in layer_config_tensors.items():
            idx = torch.tensor(actions[layer], device=confs.device, dtype=torch.long)
            chosen = confs[torch.arange(confs.size(0)), idx]
            vecs.append(chosen.mean(dim=0))
        if len(vecs) == 0:
            return torch.zeros(layer_config_tensors[0].size(-1), device=self.device)
        action_emb = torch.stack(vecs, dim=0).mean(dim=0)
        return action_emb

    def _soft_update(self, online, target):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def update(self, batch, layer_config_tensors):
        obs_batch = torch.stack(batch.obs).to(self.device)
        next_obs_batch = torch.stack(batch.next_obs).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)
        done_batch = torch.stack(batch.done).to(self.device)
        B, N, D = obs_batch.shape
        self.dagnn.train()
        self.actor.train()
        self.critic.train()
        q1_list = []
        q2_list = []
        target_q_list = []
        for b in range(B):
            obs = obs_batch[b]
            next_obs = next_obs_batch[b]
            node_embs, g_emb = self.dagnn(self.g, obs)
            layer_embs = split_expert_embeddings_by_layer(self.g, node_embs)
            actions, _ = self.actor(layer_embs, g_emb, layer_config_tensors)
            action_emb = self._build_action_embedding(layer_config_tensors, actions)
            pooled = node_embs.mean(dim=0)
            q1, q2 = self.critic(g_emb, pooled, action_emb)
            q1_list.append(q1)
            q2_list.append(q2)
            with torch.no_grad():
                n_embs, n_g = self.target_dagnn(self.g, next_obs)
                n_layer_embs = split_expert_embeddings_by_layer(self.g, n_embs)
                n_actions, _ = self.target_actor(n_layer_embs, n_g, layer_config_tensors)
                n_action_emb = self._build_action_embedding(layer_config_tensors, n_actions)
                n_pooled = n_embs.mean(dim=0)
                tq1, tq2 = self.target_critic(n_g, n_pooled, n_action_emb)
                tq = torch.min(tq1, tq2)
                target = reward_batch[b] + self.gamma * (1.0 - done_batch[b]) * tq
                target_q_list.append(target)
        q1_tensor = torch.stack(q1_list)
        q2_tensor = torch.stack(q2_list)
        target_tensor = torch.stack(target_q_list).detach()
        critic_loss = ((q1_tensor - target_tensor) ** 2 + (q2_tensor - target_tensor) ** 2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        actor_loss_vals = []
        behavior_penalty_vals = []
        for b in range(B):
            obs = obs_batch[b]
            node_embs, g_emb = self.dagnn(self.g, obs)
            layer_embs = split_expert_embeddings_by_layer(self.g, node_embs)
            actions, logits = self.actor(layer_embs, g_emb, layer_config_tensors)
            action_emb = self._build_action_embedding(layer_config_tensors, actions)
            pooled = node_embs.mean(dim=0)
            q1_pi, q2_pi = self.critic(g_emb, pooled, action_emb)
            q_pi = torch.min(q1_pi, q2_pi)
            avg_logit = []
            for l in logits:
                avg_logit.append(logits[l].mean())
            if len(avg_logit) > 0:
                behavior_ref = torch.stack(avg_logit).mean()
                behavior_penalty = (behavior_ref ** 2)
            else:
                behavior_penalty = torch.tensor(0.0, device=self.device)
            loss_pi = -q_pi + self.behavior_reg * behavior_penalty
            actor_loss_vals.append(loss_pi)
            behavior_penalty_vals.append(behavior_penalty)
        actor_loss = torch.stack(actor_loss_vals).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self._soft_update(self.dagnn, self.target_dagnn)
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}
