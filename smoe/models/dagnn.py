import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

class DAGLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.gate_linear = nn.Linear(in_dim + out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, g, h, topo_nodes, pred_map):
        out = torch.zeros_like(self.self_linear(h))
        for nid in topo_nodes:
            hv = h[nid]
            hs = self.self_linear(hv)
            preds = pred_map[nid]
            if len(preds) > 0:
                neigh = h[preds].mean(dim=0)
                hn = self.neigh_linear(neigh)
                m = hs + hn
            else:
                m = hs
            gate_input = torch.cat([hv, m], dim=-1)
            gate = torch.sigmoid(self.gate_linear(gate_input))
            v = gate * m + (1.0 - gate) * hv
            out[nid] = v
        out = self.norm(out)
        out = F.relu(out)
        return out

class DAGPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.att_linear = nn.Linear(in_dim, hidden_dim)
        self.score_linear = nn.Linear(hidden_dim, 1)

    def forward(self, g, h):
        sinks = [n for n in g.nodes() if g.out_degree(n) == 0]
        if len(sinks) == 0:
            a = torch.tanh(self.att_linear(h))
            scores = self.score_linear(a).squeeze(-1)
            w = torch.softmax(scores, dim=0)
            return (w.unsqueeze(-1) * h).sum(dim=0)
        sink_embs = h[sinks]
        a = torch.tanh(self.att_linear(sink_embs))
        scores = self.score_linear(a).squeeze(-1)
        w = torch.softmax(scores, dim=0)
        out = (w.unsqueeze(-1) * sink_embs).sum(dim=0)
        return out

class SimpleDAGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DAGLayer(hidden_dim, hidden_dim))
        self.pool = DAGPooling(hidden_dim, hidden_dim)

    def forward(self, g, node_feats):
        h = self.input_proj(node_feats)
        topo_nodes = list(nx.topological_sort(g))
        pred_map = {n: list(g.predecessors(n)) for n in g.nodes()}
        for layer in self.layers:
            h = layer(g, h, topo_nodes, pred_map)
        graph_emb = self.pool(g, h)
        return h, graph_emb
