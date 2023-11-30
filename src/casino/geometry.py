try:
    import numpy as np
except:
    import logging

    logging.debug("numpy not availble. Most functionality in latents.py will break")
    np.zeros = [0, 0]

from typing import List, Union


def circle_samples(
    radius: float = 1.0,
    center: Union["np.ndarray", List[float]] = [0.0, 0.0],
    n: int = 50,
):
    """
    Returns samples around center
    """
    center = np.array(center)
    assert center.shape == (2,)
    lin_samples = np.linspace(
        start=0.0, stop=2 * np.pi, num=n, endpoint=False, dtype=np.float32
    )
    circle_samples = (
        radius * np.stack([np.cos(lin_samples), np.sin(lin_samples)], axis=1) + center
    )
    return circle_samples
