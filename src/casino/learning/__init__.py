from . import checkpoints, default_init, learning_rate_schedulers

__all__ = ["checkpoints", "default_init", "learning_rate_schedulers"]


try:
    import torch
    DEFAULT_TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
except:
    import logging
    logging.debug(
        "torch not availble. DEFAULT_TORCH_DEVICE is not set"
    )

