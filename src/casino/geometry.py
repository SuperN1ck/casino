from typing import Union, Tuple, List

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


try:
    import torch
except:
    import logging

    logging.debug("torch not availble. Some functionality in geometry.py will break")


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
    copy=True,
):
    """
    Given a translation and rotation, this functions returns a 4x4 homogeneous transformation matrix
    """
    t = np.array(t)
    R = np.array(R)

    assert t.shape[-1:] == (3,)
    assert R.shape[-2:] == (3, 3)
    assert t.shape[:-1] == R.shape[:-2]

    B = t.shape[:-1]  # Extract batch

    trans = batched_eye_np(B, 4)
    trans[..., :3, :3] = R.copy() if copy else R
    trans[..., :3, 3] = t.copy() if copy else t
    return trans


def to_transformation_matrix_th(
    t: Union["torch.Tensor", List[float]] = [0.0, 0.0, 0.0],
    R: Union["torch.Tensor", List[List[float]]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
):
    """
    Returns in-place?
    """
    t = torch.Tensor(t)
    R = torch.Tensor(R)

    assert t.shape[-1:] == (3,)
    assert R.shape[-2:] == (3, 3)
    assert t.shape[:-1] == R.shape[:-2]

    B = t.shape[:-1]  # Extract batch

    trans = batched_eye_th(B, 4, device=t.device)
    trans[..., :3, :3] = R
    trans[..., :3, 3] = t
    return trans


def ensure_valid_rotation(R: Union["np.ndarray", List[List[float]]]):
    R = np.array(R)

    R_updated, sca = orthogonal_procrustes(np.eye(3), R)
    return R_updated


def batched_eye_np(B: int, N: int, **kwargs):
    if isinstance(B, int):
        B = (B,)
    batch_dims = (1,) * len(B)
    return np.tile(
        np.eye(N, **kwargs).reshape(*batch_dims, N, N),
        (
            B
            + (
                1,
                1,
            )
        ),
    )


def batched_eye_th(B: Union[int, Tuple[int]], N: int, **kwargs):
    if isinstance(B, int):
        B = (B,)
    batch_dims = (1,) * len(B)
    return torch.eye(N, **kwargs).view(*batch_dims, N, N).repeat(*B, 1, 1)


def numpy_batched_eye(*args, **kwargs):
    return batched_eye_np(*args, **kwargs)


def grahm_schmidt_th(v1: "torch.Tensor", v2: "torch.Tensor") -> "torch.Tensor":
    """Compute orthonormal basis from two vectors."""
    u1 = v1
    e1 = u1 / torch.norm(u1, dim=-1, keepdim=True)
    u2 = v2 - vec_projection_th(v2, e1)
    e2 = u2 / torch.norm(u2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2, dim=-1)
    rot_matrix = torch.cat(
        [e1.unsqueeze(dim=-1), e2.unsqueeze(dim=-1), e3.unsqueeze(dim=-1)], dim=-1
    )
    return rot_matrix


def vec_projection_th(v: "torch.Tensor", e: "torch.Tensor") -> "torch.Tensor":
    """Project vector v onto unit vector e."""
    proj = torch.sum(v * e, dim=-1, keepdim=True) * e
    return proj
