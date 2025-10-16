import logging
import operator
from collections.abc import Mapping, Iterable
from collections import OrderedDict
from typing import Dict, Callable, Union, Any, Iterable, Optional

try:
    import numpy as np
except:
    logging.debug(
        "numpy not availble. Some functionality in special_dicts.py will break"
    )

try:
    import torch
except:
    logging.debug(
        "torch not availble. Some functionality in special_dicts.py will break"
    )


def invert_dict(input_dict):
    """
    Careful, obviousouly full copy
    """
    return {v: k for k, v in input_dict.items()}


class AccumulatorDict(dict):
    def __init__(self, *args, accumulator=operator.add, **kwargs):
        self.accumulator = accumulator
        self.update(*args, **kwargs)
        self.steps: int = 0

    def increment_step(self):
        self.steps += 1

    def increment(self, key, val, increment_steps: bool = False):
        """
        This will increment the value for a given key
        """
        if dict.__contains__(self, key):
            val = self.accumulator(dict.__getitem__(self, key), val)
        dict.__setitem__(self, key, val)

        if increment_steps:
            self.increment_step()

    def increment_dict(self, other: Dict, increment_steps: bool = True):
        for key, value in other.items():
            # Handle the case for nested dicts
            if isinstance(value, Mapping):
                if dict.__contains__(self, key):
                    old_dict = dict.__getitem__(self, key)
                    if not isinstance(old_dict, AccumulatorDict):
                        new_dict = type(self)(accumulator=self.accumulator)
                        new_dict.increment_dict(old_dict)
                        old_dict = new_dict
                    # This performs the operation in-place
                    # TODO: Is there a clever way to verify this?
                    old_dict.increment_dict(value)
                    dict.__setitem__(self, key, old_dict)
                else:
                    # value is a dict, but not an AccumulatorDict
                    if not isinstance(value, AccumulatorDict):
                        new_dict = type(self)(accumulator=self.accumulator)
                        new_dict.increment_dict(value)
                        value = new_dict
                    dict.__setitem__(self, key, value)
            else:
                self.increment(key, value)

        if increment_steps:
            self.increment_step()

    def as_default_dict(self):
        """
        Careful, full copy
        """
        return {k: v for k, v in self.items()}


def map_nested_dicts(
    ob: Union[Any, Mapping], func: Callable = lambda x: x, inplace: bool = False
):
    if inplace:
        for key, value in ob.items():
            if isinstance(value, Mapping):
                map_nested_dicts(value, func, inplace=inplace)
            else:
                ob[key] = func(value)
        return ob
    else:
        if isinstance(ob, Mapping):
            if not isinstance(ob, Dict):
                print(
                    "Warning. Creating a new dict but the input data type is different."
                )
            return {
                key: map_nested_dicts(value, func, inplace=inplace)
                for key, value in ob.items()
            }
        else:
            return func(ob)


class NumpyConcatenateDict(AccumulatorDict):
    def __init__(self, axis: int = 0, **kwargs):
        super(NumpyConcatenateDict, self).__init__(**kwargs)

        def wrapped_concatenate(arr0: "np.ndarray", arr1: "np.ndarray"):
            return np.concatenate((arr0, arr1), axis=axis)

        self.accumulator = wrapped_concatenate


class TorchConcatenateDict(AccumulatorDict):
    def __init__(self, dim: int = 0, allow_other_datatypes: bool = True, **kwargs):
        super(TorchConcatenateDict, self).__init__(**kwargs)
        self.allow_other_datatypes = allow_other_datatypes

        def wrapped_concatenate(tensor0: "torch.Tensor", tensor1: "torch.Tensor"):
            if isinstance(tensor0, torch.Tensor) and isinstance(tensor1, torch.Tensor):
                return torch.concatenate((tensor0, tensor1), dim=dim)
            elif (
                allow_other_datatypes
                and dim == 0
                and isinstance(tensor0, Iterable)
                and isinstance(tensor1, Iterable)
            ):
                return tensor0 + tensor1
            else:
                raise ValueError(
                    f"Cannot concatenate {type(tensor0)} and {type(tensor1)}"
                )

        self.accumulator = wrapped_concatenate


class IndexDict(dict):
    """
    This will incrementally count up for unseen keys
    """

    freeze = False  # If True, the dict will not allow new keys to be added

    def __init__(self, counter_start: int = 0, *args, **kwargs):
        self.counter = counter_start
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        # TODO [NH] 2025/06/04: Enable this?
        # If passing e.g. torch.Tensor, multiple tensor elements, or other non-hashable types,
        # we need to convert the key to a common representation
        # key = repr(key)
        if dict.__contains__(self, key):
            val = dict.__getitem__(self, key)
        elif not self.freeze:
            val = self.counter
            dict.__setitem__(self, key, val)
            self.counter += 1
        else:
            raise KeyError(
                f"Key '{key}' not found in IndexDict and freeze is set to True."
            )
        return val

    # def __setitem__(self, key, val):
    #   print('SET', key, val)
    #   dict.__setitem__(self, key, val)

    # def __repr__(self):
    #   dictrepr = dict.__repr__(self)
    #   return '%s(%s)' % (type(self).__name__, dictrepr)

    # def update(self, *args, **kwargs):
    #   print('update', args, kwargs)
    #   for k, v in dict(*args, **kwargs).items():
    #     self[k] = v

    def get_key(self, index):
        for k, v in self.items():
            if v != index:
                continue

            return k


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        assert "module." in k, "Trying to convert non data parallel checkpoint!"
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


# def update_recursively(target_dict, source_dict):
#     # Update the target_dict with the values of the source dict
#     for key, value in source_dict.items():
#         if isinstance(Dict, value) and key in target_dict.keys():
#             update_recursively


def deindex(
    input_iterable: Iterable, deindex: Optional[Union[Iterable[int], int]] = None
):
    """
    Removes a list of indices from an iterable
    Careful, full copy
    """
    if isinstance(deindex, int):
        deindex = [deindex]
    # Handle cases for negative numbers
    deindex += [len(input_iterable) + idx for idx in deindex if idx < 0]
    return [x for idx, x in enumerate(input_iterable) if not idx in deindex]
