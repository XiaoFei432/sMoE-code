import argparse
import os
import csv
import yaml
import torch
from smoe.envs.moe_env import MoEGraphEnv
from smoe.agent.gprl_agent import GPRLAgent
from smoe.utils.seeding import set_seed
from scripts.train_offline import build_layer_config_candidates

def workload_slo(name):
    n = name.lower()
    if n == "bert":
        return 80.0
    if n == "gpt2":
        return 100.0
    return 120.0

def build_agent(cfg, env, device, config_dim):
    agent = GPRLAgent(
        graph=env.graph,
        obs_dim=cfg["env"]["feature_dim"],
        config_dim=config_dim,
        dagnn_hidden=cfg["dagnn"]["hidden_dim"],
        dagnn_layers=cfg["dagnn"]["num_layers"],
        pointer_hidden=cfg["pointer"]["hidden_dim"],
        siso_hidden=cfg["siso"]["hidden_dim"],
        actor_lr=cfg["rl"]["actor_lr"],
        critic_lr=cfg["rl"]["critic_lr"],
        gamma=cfg["rl"]["gamma"],
        tau=0.005,
        behavior_reg=0.01,
        device=device,
    )
    return agent

def load_checkpoint_if_exists(agent, path, device):
    if path is None:
        return False
    if not os.path.isfile(path):
        return False
    data = torch.load(path, map_location=device)
    if "dagnn" in data and "actor" in data and "critic" in data:
        agent.dagnn.load_state_dict(data["dagnn"])
        agent.actor.load_state_dict(data["actor"])
        agent.critic.load_state_dict(data["critic"])
        if hasattr(agent, "target_dagnn"):
            agent.target_dagnn.load_state_dict(data["dagnn"])
        if hasattr(agent, "target_actor"):
            agent.target_actor.load_state_dict(data["actor"])
        if hasattr(agent, "target_critic"):
            agent.target_critic.load_state_dict(data["critic"])
        return True
    return False

def run_eval_episode(env, agent, layer_config_tensors, max_steps, device, epsilon, slo_global):
    obs_np = env.reset()
    obs = torch.from_numpy(obs_np).to(device)
    costs = []
    p95_list = []
    patterns = []
    viol_flags = []
    for t in range(max_steps):
        actions = agent.select_actions(obs, layer_config_tensors, epsilon=epsilon)
        layer_action_dict = {}
        for layer, expert_ids in env.layer_expert_ids.items():
            layer_cfg = []
            for idx, nid in enumerate(expert_ids):
                a_idx = actions[layer][idx]
                c_vec = layer_config_tensors[layer][idx, a_idx].detach().cpu().numpy()
                conf = {
                    "mem": float(c_vec[0]),
                    "gpu": float(c_vec[1]),
                    "replicas": int(round(float(c_vec[2]) * env.max_replicas)),
                    "concurrency": int(round(float(c_vec[3]) * env.max_concurrency)),
                }
                if conf["replicas"] < 1:
                    conf["replicas"] = 1
                if conf["concurrency"] < 1:
                    conf["concurrency"] = 1
                layer_cfg.append(conf)
            layer_action_dict[layer] = layer_cfg
        next_obs_np, reward, info_done, info = env.step(layer_action_dict)
        next_obs = torch.from_numpy(next_obs_np).to(device)
        obs = next_obs
        cost = info["cost"]
        p95 = info["p95_latency"]
        pat = info["pattern"]
        costs.append(cost)
        p95_list.append(p95)
        patterns.append(pat)
        viol_flags.append(1.0 if p95 > slo_global else 0.0)
    avg_cost = sum(costs) / float(len(costs))
    avg_p95 = sum(p95_list) / float(len(p95_list))
    viol_rate = sum(viol_flags) / float(len(viol_flags))
    avg_margin = sum([(v - slo_global) for v in p95_list]) / float(len(p95_list))
    pattern_cost = {}
    pattern_latency = {}
    pattern_viol = {}
    pattern_cnt = {}
    for c, l, v, p in zip(costs, p95_list, viol_flags, patterns):
        if p not in pattern_cost:
            pattern_cost[p] = 0.0
            pattern_latency[p] = 0.0
            pattern_viol[p] = 0.0
            pattern_cnt[p] = 0
        pattern_cost[p] += c
        pattern_latency[p] += l
        pattern_viol[p] += v
        pattern_cnt[p] += 1
    pattern_stats = {}
    for p in pattern_cost:
        n = float(pattern_cnt[p])
        pattern_stats[p] = {
            "cost": pattern_cost[p] / n,
            "p95": pattern_latency[p] / n,
            "viol": pattern_viol[p] / n,
        }
    return {
        "avg_cost": avg_cost,
        "avg_p95": avg_p95,
        "viol_rate": viol_rate,
        "avg_margin": avg_margin,
        "pattern_stats": pattern_stats,
    }

def aggregate_results(all_results):
    out = {}
    for key in all_results:
        entries = all_results[key]
        if not entries:
            continue
        s_cost = 0.0
        s_p95 = 0.0
        s_viol = 0.0
        s_margin = 0.0
        for e in entries:
            s_cost += e["avg_cost"]
            s_p95 += e["avg_p95"]
            s_viol += e["viol_rate"]
            s_margin += e["avg_margin"]
        n = float(len(entries))
        out[key] = {
            "avg_cost": s_cost / n,
            "avg_p95": s_p95 / n,
            "viol_rate": s_viol / n,
            "avg_margin": s_margin / n,
        }
    return out

def write_csv(path, records, fieldnames):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workloads", nargs="+", default=["BERT", "GPT2", "B2B"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    device = cfg["experiment"]["device"]
    records = []
    for wname in args.workloads:
        slo_global = workload_slo(wname)
        all_results = {}
        for sd in args.seeds:
            set_seed(sd)
            env = MoEGraphEnv(
                num_layers=cfg["env"]["num_layers"],
                experts_per_layer=cfg["env"]["experts_per_layer"],
                feature_dim=cfg["env"]["feature_dim"],
                max_concurrency=cfg["env"]["max_concurrency"],
                max_replicas=cfg["env"]["max_replicas"],
                workload_name=wname,
                seed=sd,
            )
            layer_config_tensors, config_dim = build_layer_config_candidates(env, device)
            agent = build_agent(cfg, env, device, config_dim)
            load_checkpoint_if_exists(agent, args.checkpoint, device)
            key = (wname, sd)
            all_results[key] = []
            for ep in range(args.episodes):
                res = run_eval_episode(env, agent, layer_config_tensors, args.max_steps, device, args.epsilon, slo_global)
                all_results[key].append(res)
                r = {
                    "workload": wname,
                    "seed": sd,
                    "episode": ep,
                    "avg_cost": res["avg_cost"],
                    "avg_p95": res["avg_p95"],
                    "viol_rate": res["viol_rate"],
                    "avg_margin": res["avg_margin"],
                }
                for pat in ["slow", "normal", "bursty"]:
                    s = res["pattern_stats"].get(pat)
                    if s is None:
                        r["cost_%s" % pat] = ""
                        r["p95_%s" % pat] = ""
                        r["viol_%s" % pat] = ""
                    else:
                        r["cost_%s" % pat] = s["cost"]
                        r["p95_%s" % pat] = s["p95"]
                        r["viol_%s" % pat] = s["viol"]
                records.append(r)
            agg = aggregate_results(all_results)
            for k in agg:
                wk, sd2 = k
                v = agg[k]
                print("workload", wk, "seed", sd2, "avg_cost", round(v["avg_cost"], 3), "avg_p95", round(v["avg_p95"], 2), "viol_rate", round(v["viol_rate"], 3), "margin", round(v["avg_margin"], 2))
    if args.output_csv is not None:
        base_fields = ["workload", "seed", "episode", "avg_cost", "avg_p95", "viol_rate", "avg_margin"]
        pats = ["slow", "normal", "bursty"]
        extra = []
        for p in pats:
            extra.append("cost_%s" % p)
            extra.append("p95_%s" % p)
            extra.append("viol_%s" % p)
        write_csv(args.output_csv, records, base_fields + extra)

if __name__ == "__main__":
    main()
