import random

import numpy as np
import torch


class RandomSeedManager:
    def __init__(self, seed_value):
        self.seed_value = seed_value

    def set_seed(self):
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_value)
            torch.cuda.manual_seed_all(self.seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("random seed set to:", self.seed_value)

    def worker_init_fn(self, worker_id):
        seed = self.seed_value + worker_id
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
