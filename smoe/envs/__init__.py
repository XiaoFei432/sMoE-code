import threading
from .moe_env import MoEGraphEnv

ENV_REGISTRY = {}
_ENV_LOCK = threading.RLock()

ENV_REGISTRY["moe_graph"] = MoEGraphEnv
ENV_REGISTRY["default"] = MoEGraphEnv


def register_env(name, cls):
    with _ENV_LOCK:
        ENV_REGISTRY[name] = cls


def get_env_class(name):
    with _ENV_LOCK:
        if name in ENV_REGISTRY:
            return ENV_REGISTRY[name]
        return ENV_REGISTRY.get("default", MoEGraphEnv)


def create_env(name="default", config=None):
    cfg = config or {}
    cls = get_env_class(name)
    num_layers = cfg.get("num_layers", 12)
    experts_per_layer = cfg.get("experts_per_layer", 6)
    feature_dim = cfg.get("feature_dim", 16)
    max_concurrency = cfg.get("max_concurrency", 10)
    max_replicas = cfg.get("max_replicas", 8)
    seed = cfg.get("seed", 42)
    workload_name = cfg.get("workload_name", "B2B")
    window_size = cfg.get("window_size", 1.0)
    env = cls(
        num_layers=num_layers,
        experts_per_layer=experts_per_layer,
        feature_dim=feature_dim,
        max_concurrency=max_concurrency,
        max_replicas=max_replicas,
        seed=seed,
        window_size=window_size,
        workload_name=workload_name,
    )
    return env
