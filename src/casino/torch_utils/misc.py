import gc
import logging

try:
    import torch
except:
    logging.debug("torch not availble. Most functionality in torch_utils will break.")


def torch_batched_eye(B: int, N: int, **kwargs):
    return torch.eye(N, **kwargs).reshape(1, N, N).repeat(B, 1, 1)


def clear_torch_memory():
    gc.collect()
    torch.cuda.empty_cache()
