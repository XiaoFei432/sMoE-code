import math
import random
import numpy as np
import torch
import networkx as nx

class LayerIndex:
    def __init__(self, graph):
        self.graph = graph
        self.layer_to_nodes = {}
        self.node_to_layer = {}
        self._build()

    def _build(self):
        for nid, data in self.graph.nodes(data=True):
            kind = data.get("kind")
            if kind != "expert":
                continue
            layer = int(data.get("layer", -1))
            if layer not in self.layer_to_nodes:
                self.layer_to_nodes[layer] = []
            self.layer_to_nodes[layer].append(int(nid))
            self.node_to_layer[int(nid)] = layer
        for l in self.layer_to_nodes:
            self.layer_to_nodes[l] = sorted(self.layer_to_nodes[l])

    def layers(self):
        return sorted(self.layer_to_nodes.keys())

    def nodes_in_layer(self, layer):
        return list(self.layer_to_nodes.get(layer, []))

    def layer_of(self, node_id):
        return self.node_to_layer.get(int(node_id), None)


def is_dag(graph):
    try:
        nx.find_cycle(graph, orientation="original")
        return False
    except nx.exception.NetworkXNoCycle:
        return True


def topo_order(graph):
    if not is_dag(graph):
        return list(graph.nodes())
    return list(nx.topological_sort(graph))


def degree_stats(graph):
    indeg = []
    outdeg = []
    for n in graph.nodes():
        indeg.append(graph.in_degree(n))
        outdeg.append(graph.out_degree(n))
    if len(indeg) == 0:
        return {"in_min": 0, "in_max": 0, "out_min": 0, "out_max": 0}
    return {
        "in_min": int(min(indeg)),
        "in_max": int(max(indeg)),
        "out_min": int(min(outdeg)),
        "out_max": int(max(outdeg)),
    }


def _ensure_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    t = torch.tensor(np.asarray(x), dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def split_expert_embeddings_by_layer(graph, node_embs, device=None):
    emb = _ensure_tensor(node_embs, device=device)
    index = LayerIndex(graph)
    out = {}
    for layer in index.layers():
        ids = index.nodes_in_layer(layer)
        if not ids:
            continue
        slices = [emb[int(n)] for n in ids]
        stacked = torch.stack(slices, dim=0)
        out[layer] = stacked
    return out


def pack_layer_action_embedding(layer_config_tensors, actions, device=None):
    vecs = []
    for layer, confs in layer_config_tensors.items():
        idx = torch.tensor(actions[layer], dtype=torch.long, device=confs.device)
        chosen = confs[torch.arange(confs.size(0)), idx]
        v = chosen.mean(dim=0)
        vecs.append(v)
    if not vecs:
        if device is None:
            return torch.zeros(1)
        return torch.zeros(1, device=device)
    t = torch.stack(vecs, dim=0).mean(dim=0)
    if device is not None:
        t = t.to(device)
    return t


def random_walk_layers(graph, start=None, length=4):
    nodes = topo_order(graph)
    if not nodes:
        return []
    if start is None:
        cur = random.choice(nodes)
    else:
        cur = start
    path = [cur]
    for _ in range(max(0, length - 1)):
        succ = list(graph.successors(cur))
        if not succ:
            break
        cur = random.choice(succ)
        path.append(cur)
    return path


def structural_hash(graph):
    nodes = sorted(list(graph.nodes()))
    edges = sorted(list(graph.edges()))
    data = []
    for n in nodes:
        d = graph.nodes[n]
        layer = d.get("layer", -1)
        kind = d.get("kind", "")
        data.append("N:%s:%s:%s" % (str(n), str(layer), str(kind)))
    for u, v in edges:
        data.append("E:%s:%s" % (str(u), str(v)))
    s = "|".join(data)
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h
