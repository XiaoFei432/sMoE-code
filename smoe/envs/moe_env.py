import math
import numpy as np
import networkx as nx

class MoEGraphEnv:
    def __init__(self, num_layers=12, experts_per_layer=6, feature_dim=16, max_concurrency=10, max_replicas=8, seed=42, window_size=1.0, workload_name="B2B"):
        self.num_layers = num_layers
        self.experts_per_layer = experts_per_layer
        self.feature_dim = feature_dim
        self.max_concurrency = max_concurrency
        self.max_replicas = max_replicas
        self.window_size = float(window_size)
        self.workload_name = workload_name
        self.rng = np.random.RandomState(seed)
        self.graph = self._build_dag()
        self.non_moe_nodes = [n for n, d in self.graph.nodes(data=True) if d["kind"] == "non_moe"]
        self.expert_nodes = [n for n, d in self.graph.nodes(data=True) if d["kind"] == "expert"]
        self.layer_expert_ids = self._build_layer_expert_ids()
        self.layer_slo = self._build_layer_slos()
        self.interval_index = 0
        self.cur_pattern = "bursty"
        self.node_features = self._init_node_features()
        self.layer_token_share = self._init_layer_distribution()
        self.layer_request_share = self._init_layer_distribution()
        self.layer_load_factor = self._init_layer_load_factor()

    def _build_dag(self):
        g = nx.DiGraph()
        nid = 0
        for i in range(self.num_layers + 1):
            g.add_node(nid, kind="non_mo	e", layer=i, name="p_%d" % i)
            nid += 1
        base = nid
        for l in range(self.num_layers):
            for e in range(self.experts_per_layer):
                g.add_node(nid, kind="expert", layer=l, name="e_%d_%d" % (l, e))
                g.add_edge(l, nid)
                g.add_edge(nid, l + 1)
                nid += 1
        return g

    def _build_layer_expert_ids(self):
        layer_map = {}
        for nid, d in self.graph.nodes(data=True):
            if d["kind"] == "expert":
                layer = d["layer"]
                if layer not in layer_map:
                    layer_map[layer] = []
                layer_map[layer].append(nid)
        return layer_map

    def _build_layer_slos(self):
        slo_map = {}
        base = 80.0
        if self.workload_name.lower() == "bert":
            base = 80.0
        elif self.workload_name.lower() == "gpt2":
            base = 100.0
        else:
            base = 120.0
        for l in range(self.num_layers):
            factor = 1.0 + float(l) / float(max(1, self.num_layers - 1)) * 0.2
            slo_map[l] = base * factor / float(self.num_layers)
        return slo_map

    def _init_layer_distribution(self):
        layer_map = {}
        for l in range(self.num_layers):
            alpha = 1.5 + 0.5 * self.rng.rand()
            v = self.rng.dirichlet([alpha] * self.experts_per_layer)
            layer_map[l] = v
        return layer_map

    def _init_layer_load_factor(self):
        layer_map = {}
        for l in range(self.num_layers):
            base = 1.0 + 0.2 * l
            scale = 0.5 + self.rng.rand()
            layer_map[l] = base * scale
        return layer_map

    def _init_node_features(self):
        n = self.graph.number_of_nodes()
        feats = np.zeros((n, self.feature_dim), dtype=np.float32)
        for nid, d in self.graph.nodes(data=True):
            if d["kind"] == "expert":
                layer = d["layer"]
                base_scale = 1.0 + 0.1 * layer
                cc = self.rng.uniform(0.1, 0.9)
                ru = self.rng.uniform(0.2, 0.95)
                cu = self.rng.uniform(0.1, 0.95)
                tps = self.rng.uniform(0.0, 1.0) * base_scale
                rps = self.rng.uniform(0.0, 1.0) * base_scale
                el = self.rng.uniform(0.6, 1.4)
                ts = self.layer_token_share[layer][(nid - (self.num_layers + 1)) % self.experts_per_layer]
                rs = self.layer_request_share[layer][(nid - (self.num_layers + 1)) % self.experts_per_layer]
                tre = -ts * math.log(ts + 1e-8)
                elv = self.rng.uniform(0.0, 0.25)
                sr = self.rng.uniform(0.7, 1.5)
                sf = float(layer + 1) / float(self.num_layers)
                vec = np.array([cc, ru, cu, tps, rps, el, ts, rs, tre, elv, sr, sf], dtype=np.float32)
                feats[nid, : vec.shape[0]] = vec
            else:
                depth = d["layer"]
                cc = self.rng.uniform(0.2, 0.8)
                ru = self.rng.uniform(0.25, 0.85)
                cu = self.rng.uniform(0.1, 0.7)
                el = self.rng.uniform(0.4, 1.0) * (1.0 + 0.02 * depth)
                vec = np.array([cc, ru, cu, el], dtype=np.float32)
                feats[nid, : vec.shape[0]] = vec
        return feats

    def _shift_pattern(self):
        if self.interval_index % 30 == 0:
            mode = self.interval_index // 30
            if mode % 3 == 0:
                self.cur_pattern = "slow"
            elif mode % 3 == 1:
                self.cur_pattern = "normal"
            else:
                self.cur_pattern = "bursty"

    def _refresh_layer_distributions(self):
        for l in range(self.num_layers):
            base = self.layer_load_factor[l]
            jitter = 0.3 + 0.7 * self.rng.rand()
            if self.cur_pattern == "slow":
                alpha = 2.0 + 0.2 * l
            elif self.cur_pattern == "normal":
                alpha = 1.5 + 0.5 * self.rng.rand()
            else:
                alpha = 0.9 + 0.3 * self.rng.rand()
            token_share = self.rng.dirichlet([alpha] * self.experts_per_layer)
            req_alpha = alpha + 0.4 * (self.rng.rand(self.experts_per_layer) - 0.5)
            req_alpha = np.maximum(req_alpha, 0.2)
            request_share = self.rng.dirichlet(req_alpha)
            self.layer_token_share[l] = token_share
            self.layer_request_share[l] = request_share
            self.layer_load_factor[l] = base * jitter

    def _update_node_features_from_distributions(self):
        feats = self.node_features
        for l in range(self.num_layers):
            ts_vec = self.layer_token_share[l]
            rs_vec = self.layer_request_share[l]
            for idx, nid in enumerate(self.layer_expert_ids[l]):
                ts = ts_vec[idx]
                rs = rs_vec[idx]
                tre = -ts * math.log(ts + 1e-8)
                elv = self.rng.uniform(0.0, 0.3)
                sr = 0.8 + 0.6 * abs(ts - rs)
                feats[nid, 6] = ts
                feats[nid, 7] = rs
                feats[nid, 8] = tre
                feats[nid, 9] = elv
                feats[nid, 10] = sr
        self.node_features = feats

    def _compute_layer_cost_and_latency(self, layer, expert_ids, configs, slo_ms):
        layer_cost = 0.0
        layer_lat = 0.0
        token_share = self.layer_token_share[layer]
        request_share = self.layer_request_share[layer]
        for idx, nid in enumerate(expert_ids):
            conf = configs[idx]
            mem = conf["mem"]
            gpu = conf["gpu"]
            replicas = max(1, conf["replicas"])
            conc = max(1, conf["concurrency"])
            ts = token_share[idx]
            rs = request_share[idx]
            inflow = ts * self.layer_load_factor[layer]
            base = 40.0 + 4.0 * layer
            parallel_factor = 1.0 + 0.7 * math.log(1.0 + float(replicas))
            resource_factor = 1.0 + 1.2 * mem + 1.0 * gpu
            conc_factor = 1.0 + float(max(0, conc - 4)) / 6.0
            imbalance = 1.0 + 0.5 * abs(ts - rs)
            lat = base * imbalance * conc_factor / max(0.8, resource_factor * parallel_factor)
            lat = lat * (1.0 + 0.3 * self.rng.rand())
            unit_cost = (mem * 0.5 + gpu * 0.5) * float(replicas) * (0.4 + 0.6 * inflow) + 0.1 * conc
            layer_cost += unit_cost
            if lat > layer_lat:
                layer_lat = lat
        penalty = 0.0
        if layer_lat > slo_ms:
            penalty = (layer_lat - slo_ms) * 0.2
        return layer_cost + penalty, layer_lat

    def reset(self):
        self.interval_index = 0
        self.cur_pattern = "bursty"
        self.layer_token_share = self._init_layer_distribution()
        self.layer_request_share = self._init_layer_distribution()
        self.layer_load_factor = self._init_layer_load_factor()
        self.node_features = self._init_node_features()
        return self.node_features

    def step(self, layer_actions, global_slo_ms=100.0):
        self.interval_index += 1
        self._shift_pattern()
        self._refresh_layer_distributions()
        self._update_node_features_from_distributions()
        total_cost = 0.0
        max_lat = 0.0
        for l, expert_ids in self.layer_expert_ids.items():
            configs = layer_actions[l]
            slo_layer = self.layer_slo[l]
            layer_cost, layer_lat = self._compute_layer_cost_and_latency(l, expert_ids, configs, slo_layer)
            total_cost += layer_cost
            if layer_lat > max_lat:
                max_lat = layer_lat
        reward = -total_cost
        if max_lat > global_slo_ms:
            reward += (global_slo_ms - max_lat) * 0.5
        noise = self.rng.normal(0.0, 0.015, size=self.node_features.shape)
        self.node_features = np.clip(self.node_features + noise, 0.0, 3.0)
        info = {"cost": float(total_cost), "p95_latency": float(max_lat), "pattern": self.cur_pattern}
        return self.node_features, reward, False, info
