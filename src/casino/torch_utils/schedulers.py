import logging
from typing import Iterable, List
import functools

try:
    import torch
except:
    logging.debug("torch not availble. Most functionality in torch_utils will break.")

try:
    import diffusers
except:
    logging.debug(
        "diffusers not available. Some torch learning rate schedulers will not be available."
    )

# We follow:
# https://github.com/huggingface/transformers/blob/cbc6716945cff1d8e124d344ba0150e6e27f8b6e/src/transformers/optimization.py#L140

try:
    # This is very Frankenstein but seems to work :)
    class CosineAnnealingLinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
        def __new__(
            cls,
            optimizer: "torch.optim.Optimizer",
            T_max,
            num_warmup_epochs: int = 2,
            eta_min: float = 0.0,
            last_epoch: int = -1,
        ):

            if eta_min != 0.0:
                print(
                    f"CosineAnnealingLinearWarmupLR does not support other start and ends than 0. We are given {eta_min =}"
                )

            return diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_epochs,
                num_training_steps=T_max,
                last_epoch=last_epoch,
            )

except:
    logging.debug("Could not define CosineAnnealingLinearWarmupLR.")
