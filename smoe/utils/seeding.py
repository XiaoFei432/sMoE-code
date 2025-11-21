import os
import time
import random
import numpy as np
import torch

class SeedController:
    def __init__(self, base_seed=None):
        if base_seed is None:
            base_seed = int(time.time() * 1000) & 0xFFFFFFFF
        self.base_seed = int(base_seed)
        self.counter = 0

    def next_seed(self):
        self.counter += 1
        v = self.base_seed + self.counter * 977
        return int(v & 0x7FFFFFFF)

    def fork(self, n):
        out = []
        for _ in range(n):
            out.append(self.next_seed())
        return out


_GLOBAL_SEEDER = SeedController()


def set_seed(seed):
    s = int(seed)
    _GLOBAL_SEEDER.base_seed = s
    _GLOBAL_SEEDER.counter = 0
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    h = str(s)
    os.environ["PYTHONHASHSEED"] = h


def next_seed():
    v = _GLOBAL_SEEDER.next_seed()
    return v


def fork_seeds(n):
    return _GLOBAL_SEEDER.fork(n)
