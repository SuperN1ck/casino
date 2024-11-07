import random
import os


def set_seed(seed: int, torch_deterministic: bool = False):
    """
    Sets seed for python as well as for torch and numpy if available
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if torch_deterministic:
            # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    except:
        pass

    try:
        import numpy as np

        np.random.seed(seed)
    except:
        pass
