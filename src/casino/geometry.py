try:
    import numpy as np
except:
    import logging

    logging.debug("numpy not availble. Most functionality in geometry.py will break")
    np.zeros = [0, 0]

try:
    from scipy.linalg import orthogonal_procrustes
except:
    import logging

    logging.debug("scipy not availble. Most functionality in geometry.py will break")


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


def to_transformation_matrix(
    t: Union["np.ndarray", List[float]] = [0.0, 0.0, 0.0],
    R: Union["np.ndarray", List[List[float]]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
):
    """
    Given a translation and rotation, this functions returns a 4x4 homogeneous transformation matrix
    """
    t = np.array(t)
    R = np.array(R)

    assert t.shape == (3,)
    assert R.shape == (3, 3)

    trans = np.eye(4)
    trans[:3, :3] = R.copy()
    trans[:3, 3] = t.copy()
    return trans


def ensure_valid_rotation(R: Union["np.ndarray", List[List[float]]]):
    R = np.array(R)

    R_updated, sca = orthogonal_procrustes(np.eye(3), R)
    return R_updated
