import copy
import dataclasses
import enum
import logging
import pathlib
from typing import Any, Dict

try:
    import torch
except:
    logging.debug("torch not availble. Some functionality in dataclasses.py will break")


def default_field(obj):
    return dataclasses.field(default_factory=lambda: copy.copy(obj))


def collate_flat_dataclass_torch(dps: Any, device: str = "cpu"):
    """
    Can't process a full tree, only a single level
    TODO Make this recursive eventually
    """
    # if dataclasses.is_dataclass(dps[0]):
    # ^ Could be useful if we move to a recursive approach eventually

    collated_dp = copy.deepcopy(dps[0])
    for d_field in dataclasses.fields(dps[0]):
        # Check for None in batch
        if max(is_none:=[getattr(dp, d_field.name) is None for dp in dps]) == True:
            # Only some are None
            if min(is_none) == False:
                raise ValueError(f'Spurrious Nones found in field {d_field.name}. Either the entire batch needs to be None, or none.')
            setattr(collated_dp, d_field.name, None)
            continue

        # Use pytorch default collating for "leaves"
        collated_container = torch.utils.data.default_collate(
            [getattr(dp, d_field.name) for dp in dps]
        )
        # Move to
        if type(collated_container) == torch.Tensor:
            collated_container = collated_container.to(device)

        setattr(
            collated_dp,
            d_field.name,
            collated_container,
        )
    return collated_dp


def transform_dict(config_dict: Dict, expand: bool = True) -> Dict:
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs

    Shamelessly stolen from Brent
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        elif isinstance(v, (enum.Enum, pathlib.Path)):
            ret[k] = v.__str__()
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, "__name__") else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


def transform_dataclass(config_class: Any, expand: bool = True) -> Dict:
    return transform_dict(dataclasses.asdict(config_class), expand=expand)
