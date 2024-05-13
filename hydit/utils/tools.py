import random

import numpy as np
import torch


def set_seeds(seed_list, device=None):
    if isinstance(seed_list, (tuple, list)):
        seed = sum(seed_list)
    else:
        seed = seed_list
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return torch.Generator(device).manual_seed(seed)
