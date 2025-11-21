from .dagnn import SimpleDAGNN
from .pointer_actor import PointerActor
from .critic import DoubleCritic

MODEL_REGISTRY = {}

MODEL_REGISTRY["dagnn"] = SimpleDAGNN
MODEL_REGISTRY["pointer_actor"] = PointerActor
MODEL_REGISTRY["double_critic"] = DoubleCritic


def get_model(name):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    return None


def build_encoder(name="dagnn", in_dim=16, hidden_dim=64, num_layers=3):
    cls = get_model(name)
    if cls is None:
        cls = SimpleDAGNN
    encoder = cls(in_dim, hidden_dim, num_layers)
    return encoder


def build_actor(emb_dim, config_dim, pointer_hidden=64, siso_hidden=128):
    cls = get_model("pointer_actor")
    actor = cls(emb_dim, config_dim, pointer_hidden=pointer_hidden, siso_hidden=siso_hidden)
    return actor


def build_critic(emb_dim, config_dim, hidden_dim=128):
    cls = get_model("double_critic")
    critic = cls(emb_dim, config_dim, hidden_dim=hidden_dim)
    return critic
