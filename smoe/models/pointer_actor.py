import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerActor(nn.Module):
    def __init__(self, emb_dim, config_dim, pointer_hidden=64, siso_hidden=128, heads=2, temperature=1.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.config_dim = config_dim
        self.heads = heads
        self.temperature = temperature
        self.query_linears = nn.ModuleList()
        self.key_linears = nn.ModuleList()
        self.value_linears = nn.ModuleList()
        for _ in range(heads):
            self.query_linears.append(nn.Linear(emb_dim, pointer_hidden))
            self.key_linears.append(nn.Linear(emb_dim, pointer_hidden))
            self.value_linears.append(nn.Linear(emb_dim, pointer_hidden))
        self.merge_linear = nn.Linear(pointer_hidden * heads, emb_dim)
        self.config_gate = nn.Linear(emb_dim + config_dim, emb_dim)
        self.siso = nn.Sequential(
            nn.Linear(emb_dim * 2 + config_dim, siso_hidden),
            nn.ReLU(),
            nn.Linear(siso_hidden, siso_hidden),
            nn.ReLU(),
            nn.Linear(siso_hidden, 1)
        )

    def _multihead_context(self, expert_embs):
        k = expert_embs.size(0)
        ctx_list = []
        mean_emb = expert_embs.mean(dim=0, keepdim=True)
        for i in range(self.heads):
            q = self.query_linears[i](mean_emb)
            keys = self.key_linears[i](expert_embs)
            vals = self.value_linears[i](expert_embs)
            scores = (q @ keys.transpose(0, 1)).squeeze(0)
            scores = scores / max(1.0, float(self.temperature))
            weights = torch.softmax(scores, dim=-1)
            c = weights.unsqueeze(-1) * vals
            ctx = c.sum(dim=0)
            ctx_list.append(ctx)
        concat = torch.cat(ctx_list, dim=-1)
        out = self.merge_linear(concat)
        return out

    def forward(self, layer_expert_embs, graph_emb, layer_configs, slo_targets=None):
        actions = {}
        logits = {}
        for layer, embs in layer_expert_embs.items():
            confs = layer_configs[layer]
            k, m, cdim = confs.shape
            ctx = self._multihead_context(embs)
            layer_logits = []
            chosen = []
            for idx in range(k):
                h_e = embs[idx]
                base_state = torch.cat([h_e, ctx, graph_emb], dim=-1)
                conf_vecs = confs[idx]
                state_rep = base_state.unsqueeze(0).repeat(m, 1)
                gate_input = torch.cat([state_rep, conf_vecs], dim=-1)
                gated = torch.tanh(self.config_gate(gate_input))
                siso_in = torch.cat([gated, conf_vecs], dim=-1)
                score_vec = self.siso(siso_in).squeeze(-1)
                if slo_targets is not None:
                    adjust = torch.zeros_like(score_vec)
                    score_vec = score_vec + adjust
                prob = torch.softmax(score_vec, dim=-1)
                idx_a = torch.argmax(prob).item()
                layer_logits.append(score_vec.unsqueeze(0))
                chosen.append(idx_a)
            logits[layer] = torch.cat(layer_logits, dim=0)
            actions[layer] = chosen
        return actions, logits
