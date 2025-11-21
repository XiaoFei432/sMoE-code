import threading
from .gprl_agent import GPRLAgent

AGENT_REGISTRY = {}
_AGENT_LOCK = threading.RLock()

AGENT_REGISTRY["gprl"] = GPRLAgent
AGENT_REGISTRY["default"] = GPRLAgent


def register_agent(name, cls):
    with _AGENT_LOCK:
        AGENT_REGISTRY[name] = cls


def get_agent_class(name):
    with _AGENT_LOCK:
        if name in AGENT_REGISTRY:
            return AGENT_REGISTRY[name]
        return AGENT_REGISTRY.get("default", GPRLAgent)


def create_agent(name="gprl", graph=None, obs_dim=16, device="cpu", extra_config=None):
    cfg = extra_config or {}
    cls = get_agent_class(name)
    config_dim = cfg.get("config_dim", 4)
    dagnn_hidden = cfg.get("dagnn_hidden", 64)
    dagnn_layers = cfg.get("dagnn_layers", 3)
    pointer_hidden = cfg.get("pointer_hidden", 64)
    siso_hidden = cfg.get("siso_hidden", 128)
    actor_lr = cfg.get("actor_lr", 3e-4)
    critic_lr = cfg.get("critic_lr", 3e-4)
    gamma = cfg.get("gamma", 0.99)
    tau = cfg.get("tau", 0.005)
    behavior_reg = cfg.get("behavior_reg", 0.01)
    agent = cls(
        graph=graph,
        obs_dim=obs_dim,
        config_dim=config_dim,
        dagnn_hidden=dagnn_hidden,
        dagnn_layers=dagnn_layers,
        pointer_hidden=pointer_hidden,
        siso_hidden=siso_hidden,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        behavior_reg=behavior_reg,
        device=device,
    )
    return agent
