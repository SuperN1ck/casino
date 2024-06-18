import random


def set_seed(seed: int):
    """
    Sets seed for python as well as for torch and numpy if available
    """
    random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass

    try:
        import numpy as np

        np.random.seed(seed)
    except:
        pass
