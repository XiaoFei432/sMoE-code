import time
import random
import numpy as np
import torch
from .graph_utils import split_expert_embeddings_by_layer
from .seeding import set_seed as _set_seed

__all__ = [
    "set_seed",
    "split_expert_embeddings_by_layer",
    "timestamp_ms",
    "random_choice",
]

def set_seed(seed):
    s = int(seed)
    _set_seed(s)


def timestamp_ms():
    return int(time.time() * 1000.0)


def random_choice(seq):
    n = len(seq)
    if n == 0:
        return None
    idx = random.randrange(n)
    return seq[idx]


def random_tensor(shape, device="cpu"):
    arr = np.random.randn(*shape).astype("float32")
    t = torch.from_numpy(arr).to(device)
    return t
