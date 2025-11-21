import importlib
import threading
import copy

from .envs import ENV_REGISTRY, create_env
from .agent import AGENT_REGISTRY, create_agent
from .models import MODEL_REGISTRY
from .utils import set_seed

__all__ = [
    "ENV_REGISTRY",
    "AGENT_REGISTRY",
    "MODEL_REGISTRY",
    "create_env",
    "create_agent",
    "build_default_components",
    "set_seed",
]

_LOCK = threading.RLock()
_GLOBAL_CONFIG = {}
_COMPONENT_CACHE = {}


def _deep_merge(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return copy.deepcopy(b)
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        if k in a and k in b:
            out[k] = _deep_merge(a[k], b[k])
        elif k in a:
            out[k] = copy.deepcopy(a[k])
        else:
            out[k] = copy.deepcopy(b[k])
    return out


def set_global_config(cfg):
    global _GLOBAL_CONFIG
    with _LOCK:
        if not isinstance(cfg, dict):
            _GLOBAL_CONFIG = {}
        else:
            _GLOBAL_CONFIG = copy.deepcopy(cfg)
        _COMPONENT_CACHE.clear()


def get_global_config():
    with _LOCK:
        return copy.deepcopy(_GLOBAL_CONFIG)


def build_default_components(name, device="cpu"):
    with _LOCK:
        key = (name, device)
        if key in _COMPONENT_CACHE:
            env, agent = _COMPONENT_CACHE[key]
            return env, agent
        base_cfg = _GLOBAL_CONFIG.get("base", {})
        env_cfg = _GLOBAL_CONFIG.get("env", {})
        agent_cfg = _GLOBAL_CONFIG.get("rl", {})
        merged_env_cfg = _deep_merge(base_cfg, env_cfg)
        merged_agent_cfg = _deep_merge(base_cfg, agent_cfg)
        env = create_env(name=name, config=merged_env_cfg)
        agent = create_agent(name="gprl", graph=env.graph, obs_dim=merged_env_cfg.get("feature_dim", 16), device=device, extra_config=merged_agent_cfg)
        _COMPONENT_CACHE[key] = (env, agent)
        return env, agent


def dynamic_import(path):
    parts = path.split(":")
    if len(parts) != 2:
        return None
    mod, attr = parts
    try:
        m = importlib.import_module(mod)
    except ImportError:
        return None
    if not hasattr(m, attr):
        return None
    return getattr(m, attr)
