import gc

try:
    import torch
except:
    import logging

    logging.debug("torch not availble. Some functionality in hardwear.py will break")


def clear_torch_memory():
    gc.collect()
    torch.cuda.empty_cache()
