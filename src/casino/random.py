import random
import os

from typing import Union, List, Optional

try:
    import numpy as np
except ImportError:
    import logging

    logging.debug("numpy not available. Some functionality in random.py will break")


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


def sample_in_circle(
    inner_radius: float = 0.0,
    outer_radius: float = 1.0,
    center: Union[List, "np.ndarray"] = [0.0, 0.0],
    n: int = 1,
    rng: Optional["np.random.Generator"] = None,
) -> "np.ndarray":
    """
    Samples points uniformly in a circle with given inner and outer radius and center.
    """
    # TODO [NH] 2025/06/11: Extend this to higher-dimensional cases, e.g. by using hyperspheres or spherical coordinates.
    if center is None:
        center = [0.0, 0.0]
    center = np.array(center)

    assert center.shape == (2,)

    if rng is None:
        rng = np.random.default_rng()

    angles = rng.uniform(0, 2 * np.pi, n)
    r = rng.uniform(inner_radius, outer_radius, n)

    x = r * np.cos(angles) + center[0]
    y = r * np.sin(angles) + center[1]

    return np.column_stack((x, y))
