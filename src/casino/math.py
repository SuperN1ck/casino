import logging

try:
    import numpy as np
except:
    logging.debug("numpy not availble. Most functionality in math.py will break")

    class np:
        nan = float("nan")


try:
    import torch
except:
    logging.debug("torch not available. Some functionality in math.py will break")


from typing import List, Union


def within_range(
    value: Union["np.ndarray", List, int, float], range: Union["np.ndarray", List]
):
    """
    Performs a range check using [range[0], range[1]] (including borders)
    desired use cases:
        - simple case:
            - value=1, range=[0, 2] --> true
        - complexer cases
            - value=[1, 3], range=[0, 2] --> [true, false]
            - value=1, range=[[0, 2], [3, 5]] --> [true, false]
        - most complex, (not doing combinatorial)
            - value=[1, 3], range[[0, 2], [3, 5]] --> [true, true]

    """
    if isinstance(range, List):
        range = np.array(range)
    if not isinstance(value, np.ndarray):
        value = np.array(value)
    assert range.shape[-1] == 2
    return np.logical_and(value >= range[..., 0], value <= range[..., 1])


def multiply_along_axis(A: "np.ndarray", B: "np.ndarray", axis: int):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)


def take_along_axis(
    A: "np.ndarray", indices: "np.ndarray", oob_value: float = np.nan, axis: int = -1
):
    assert A.ndim - 1 == indices.shape[1]

    if axis == -1:
        axis = A.ndim - 1

    raise NotImplementedError()
