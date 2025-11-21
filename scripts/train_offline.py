import argparse
import itertools
import yaml
import torch
from smoe.envs.moe_env import MoEGraphEnv
from smoe.agent.gprl_agent import GPRLAgent
from smoe.utils.seeding import set_seed

def build_layer_config_candidates(env, device):
    mems = [0.2, 0.4, 0.6, 0.8, 1.0]
    gpus = [0.0, 0.25, 0.5, 0.75, 1.0]
    reps = [1, 2, 4, 6, 8]
    concs = [1, 2, 4, 6, 8, 10]
    configs = []
    for m, g, r, c in itertools.product(mems, gpus, reps, concs):
        configs.append([m, g, float(r) / float(env.max_replicas), float(c) / float(env.max_concurrency)])
    cfg_arr = torch.tensor(configs, dtype=torch.float32, device=device)
    layer_configs = {}
    for layer, expert_ids in env.layer_expert_ids.items():
        k = len(expert_ids)
        m = cfg_arr.size(0)
        layer_configs[layer] = cfg_arr.unsqueeze(0).repeat(k, 1, 1)
    return layer_configs, cfg_arr.size(-1)

def run_episode(env, agent, layer_config_tensors, max_steps, epsilon, device):
    obs_np = env.reset()
    obs = torch.from_numpy(obs_np).to(device)
    rewards = []
    costs = []
    latencies = []
    patterns = []
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
        next_obs_np, reward, done, info = env.step(layer_action_dict)
        next_obs = torch.from_numpy(next_obs_np).to(device)
        agent.replay.push(obs.cpu(), torch.zeros(1), torch.tensor(reward, dtype=torch.float32), next_obs.cpu(), torch.tensor(done, dtype=torch.float32))
        obs = next_obs
        rewards.append(reward)
        costs.append(info["cost"])
        latencies.append(info["p95_latency"])
        patterns.append(info["pattern"])
    return rewards, costs, latencies, patterns

def main(config_path):
    cfg = yaml.safe_load(open(config_path, "r"))
    set_seed(cfg["experiment"]["seed"])
    device = cfg["experiment"]["device"]
    env = MoEGraphEnv(
        num_layers=cfg["env"]["num_layers"],
        experts_per_layer=cfg["env"]["experts_per_layer"],
        feature_dim=cfg["env"]["feature_dim"],
        max_concurrency=cfg["env"]["max_concurrency"],
        max_replicas=cfg["env"]["max_replicas"],
        workload_name="B2B",
    )
    layer_config_tensors, config_dim = build_layer_config_candidates(env, device)
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
    max_steps = cfg["rl"]["max_steps"]
    batch_size = cfg["rl"]["batch_size"]
    warmup_steps = cfg["rl"]["warmup_steps"]
    log_interval = cfg["logging"]["log_interval"]
    total_intervals = 0
    best_score = None
    for outer in range(1000000):
        rewards, costs, latencies, patterns = run_episode(env, agent, layer_config_tensors, 20, epsilon=0.2, device=device)
        for i in range(len(rewards)):
            total_intervals += 1
            if len(agent.replay) > warmup_steps:
                batch = agent.replay.sample(batch_size)
                stats = agent.update(batch, layer_config_tensors)
                if total_intervals % log_interval == 0:
                    avg_reward = sum(rewards) / float(len(rewards))
                    avg_cost = sum(costs) / float(len(costs))
                    avg_latency = sum(latencies) / float(len(latencies))
                    v = "slow"
                    if patterns[-1] == "normal":
                        v = "normal"
                    elif patterns[-1] == "bursty":
                        v = "bursty"
                    print("interval", total_intervals, "pattern", v, "reward", round(avg_reward, 3), "cost", round(avg_cost, 3), "p95", round(avg_latency, 2), "actor", round(stats["actor_loss"], 4), "critic", round(stats["critic_loss"], 4))
                    score = avg_reward
                    if best_score is None or score > best_score:
                        best_score = score
            else:
                if total_intervals % log_interval == 0:
                    avg_reward = sum(rewards) / float(len(rewards))
                    avg_cost = sum(costs) / float(len(costs))
                    avg_latency = sum(latencies) / float(len(latencies))
                    print("interval", total_intervals, "warmup", "reward", round(avg_reward, 3), "cost", round(avg_cost, 3), "p95", round(avg_latency, 2))
            if total_intervals >= max_steps:
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
