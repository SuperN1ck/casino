from typing import Union, Tuple, List, Optional, Iterable

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

try:
    import roma
except:
    import logging

    logging.debug("roma not availble. Some functionality in geometry.py will break")


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

    trans = batched_eye_np(B, 4, dtype=t.dtype)
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


def batched_eye_np(B: Union[int, Iterable[int]], N: int, **kwargs):
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


def batched_eye_th(B: Union[int, Iterable[int]], N: int, **kwargs):
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


def transform_flat_pose_vector_th(
    T: "torch.Tensor",
    v: "torch.Tensor",
    pre_multiply: bool = True,
) -> "torch.Tensor":
    """
    Converts v into a 4x4 transformation matrix and applies M to it,
    then flattens it again
    """
    assert v.shape[-1] == 6, f"Expected last dimension of v to be 6, got: {v.shape}"
    v_xyz = v[..., :3]
    v_rot = roma.rotvec_to_rotmat(v[..., 3:6])

    V = to_transformation_matrix_th(t=v_xyz, R=v_rot)
    assert T.shape[-2:] == (
        4,
        4,
    ), f"T should be a 4x4 transformation matrix, but got shape: {T.shape}"
    assert V.shape[-2:] == (
        4,
        4,
    ), f"V should be a 4x4 transformation matrix, but got shape: {V.shape}"

    if pre_multiply:
        transformed_V = torch.matmul(T, V)
    else:
        transformed_V = torch.matmul(V, T)

    transformed_V_xyz = transformed_V[..., :3, 3]
    transformed_V_rot = roma.rotmat_to_rotvec(transformed_V[..., :3, :3])
    transformed_v = torch.cat([transformed_V_xyz, transformed_V_rot], dim=-1)

    # if additionally_return_transformed_matrix:
    #     return transformed_v, transformed_V

    return transformed_v


def transform_flat_pose_vector_np(
    T: "np.ndarray",
    v: "np.ndarray",
    pre_multiply: bool = True,
) -> "np.ndarray":
    """
    Converts v into a 4x4 transformation matrix and applies M to it,
    then flattens it again
    """
    assert v.shape[-1] == 6, f"Expected last dimension of v to be 6, got: {v.shape}"
    v_xyz = v[..., :3]
    v_rot = rotvec_to_rotmat_np(v[..., 3:6])

    V = to_transformation_matrix(t=v_xyz, R=v_rot)
    assert T.shape[-2:] == (
        4,
        4,
    ), f"T should be a 4x4 transformation matrix, but got shape: {T.shape}"
    assert V.shape[-2:] == (
        4,
        4,
    ), f"V should be a 4x4 transformation matrix, but got shape: {V.shape}"

    if pre_multiply:
        transformed_V = np.matmul(T, V)
    else:
        transformed_V = np.matmul(V, T)

    transformed_V_xyz = transformed_V[..., :3, 3]
    transformed_V_rot = rotmat_to_rotvec_np(transformed_V[..., :3, :3])
    transformed_v = np.concatenate([transformed_V_xyz, transformed_V_rot], axis=-1)

    # if additionally_return_transformed_matrix:
    #     return transformed_v, transformed_V

    return transformed_v


def rotvec_to_rotmat_np(rotvec: "np.ndarray") -> "np.ndarray":
    """
    Converts axis-angle representation to rotation matrix
    Args:
        rotvec (np.ndarray): Axis-angle representation of shape (..., 3)
    Returns:
        rotmat (np.ndarray): Rotation matrix of shape (..., 3, 3)
    """
    angles = np.linalg.norm(rotvec, axis=-1)
    axis = rotvec / angles[..., None]

    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    one_minus_cos = 1.0 - cos_angles

    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    rotmat = np.zeros(rotvec.shape[:-1] + (3, 3), dtype=rotvec.dtype)

    rotmat[..., 0, 0] = cos_angles + x * x * one_minus_cos
    rotmat[..., 0, 1] = x * y * one_minus_cos - z * sin_angles
    rotmat[..., 0, 2] = x * z * one_minus_cos + y * sin_angles

    rotmat[..., 1, 0] = y * x * one_minus_cos + z * sin_angles
    rotmat[..., 1, 1] = cos_angles + y * y * one_minus_cos
    rotmat[..., 1, 2] = y * z * one_minus_cos - x * sin_angles

    rotmat[..., 2, 0] = z * x * one_minus_cos - y * sin_angles
    rotmat[..., 2, 1] = z * y * one_minus_cos + x * sin_angles
    rotmat[..., 2, 2] = cos_angles + z * z * one_minus_cos

    # Handle zero-rotation case
    rotmat = np.where(
        np.isclose(angles, 0.0)[..., np.newaxis, np.newaxis],
        np.eye(3, dtype=rotvec.dtype),
        rotmat,
    )

    return rotmat


def rotmat_to_rotvec_np(rotmat: "np.ndarray") -> "np.ndarray":
    """
    Converts rotation matrix to axis-angle representation
    Args:
        rotmat (np.ndarray): Rotation matrix of shape (..., 3, 3)
    Returns:
        rotvec (np.ndarray): Axis-angle representation of shape (..., 3)
    """
    trace = np.trace(rotmat, axis1=-2, axis2=-1)
    angles = np.arccos((trace - 1) / 2.0)

    rx = rotmat[..., 2, 1] - rotmat[..., 1, 2]
    ry = rotmat[..., 0, 2] - rotmat[..., 2, 0]
    rz = rotmat[..., 1, 0] - rotmat[..., 0, 1]
    axis = np.stack([rx, ry, rz], axis=-1)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / axis_norm

    rotvec = axis * angles[..., np.newaxis]

    # Handle zero-rotation case
    rotvec = np.where(
        np.isclose(angles, 0.0)[..., np.newaxis],
        np.zeros_like(rotvec),
        rotvec,
    )

    return rotvec


def transform_sixdof_pose_vector_th(
    T: "torch.Tensor",
    v: "torch.Tensor",
    pre_multiply: bool = True,
) -> "torch.Tensor":
    """
    Converts v into a 4x4 transformation matrix and applies M to it,
    then flattens it again
    """
    assert v.shape[-1] == 9, f"Expected last dimension of v to be 9, got: {v.shape}"
    v_xyz = v[..., :3]
    v_rot = roma.special_gramschmidt(torch.stack([v[..., 3:6], v[..., 6:9]], dim=-1))

    V = to_transformation_matrix_th(t=v_xyz, R=v_rot)
    assert T.shape[-2:] == (
        4,
        4,
    ), f"T should be a 4x4 transformation matrix, but got shape: {T.shape}"
    assert V.shape[-2:] == (
        4,
        4,
    ), f"V should be a 4x4 transformation matrix, but got shape: {V.shape}"

    if pre_multiply:
        transformed_V = torch.matmul(T, V)
    else:
        transformed_V = torch.matmul(V, T)

    transformed_V_xyz = transformed_V[..., :3, 3]
    transformed_v_1 = transformed_V[..., :3, 0]
    transformed_v_2 = transformed_V[..., :3, 1]

    transformed_v = torch.cat(
        [transformed_V_xyz, transformed_v_1, transformed_v_2], dim=-1
    )

    # if additionally_return_transformed_matrix:
    #     return transformed_v, transformed_V

    return transformed_v


def axisangle2quat_np(vec):
    """
    Copied from robosuite but extedned to batches

    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angles
    angles = np.linalg.norm(vec, axis=-1, keepdims=True)

    # make sure that axis is a unit vector
    axis = vec / angles

    q = np.zeros((*vec.shape[:-1], 4))
    q[..., 3] = np.cos(angles / 2.0).squeeze(-1)  # Remove last dimension
    q[..., :3] = axis * np.sin(angles / 2.0)

    # handle zero-rotation case
    return np.where(
        ~np.isclose(angles, 0.0),
        q,  # If we are not close to 0, we will use our calculated quaternions
        np.array([0.0, 0.0, 0.0, 1.0]),  # If the angle is close we will use unit quat
    )


def linear_interpolate_np(
    start: Union["np.ndarray", List[float]],
    end: Union["np.ndarray", List[float]],
    alphas: Optional[Union["np.ndarray", List[float]]] = None,
    steps: Optional[int] = None,
) -> Union["np.ndarray", List[float]]:
    """
    Linearly interpolate between points.
    Args:
        start (Union[np.ndarray, List[float]]): Starting point.
        end (Union[np.ndarray, List[float]]): Ending point.
        alpha (Union[np.ndarray, List[float]]): Interpolation factor(s) (0.0 to 1.0).
    Returns:
        interpolated (Union[np.ndarray, List[float]]): Interpolated point.
    """
    start = np.array(start)
    end = np.array(end)
    assert start.shape == end.shape, "Start and end must have the same shape."
    assert (alphas is not None) or (
        steps is not None
    ), "Either alpha or steps must be provided."

    if alphas is None:
        assert steps is not None, "Steps must be provided if alpha is None."
        alphas = np.linspace(0, 1, steps).reshape((-1,) + (1,) * (start.ndim))

    return _linear_interpolate(
        start=start[np.newaxis, ...], end=end[np.newaxis, ...], alphas=alphas
    )


def linear_interpolate_th(
    start: Union["torch.Tensor"],
    end: Union["torch.Tensor"],
    alphas: Optional[Union["torch.Tensor", List[float]]] = None,
    steps: Optional[int] = None,
) -> Union["torch.Tensor", List[float]]:
    """
    If using steps, this function will linearly interpolate between points along the first dimension.
    Consider permuting or reshaping the tensor to fit your needs.

    Please consider reshaping your tensors to have a time dimension and directly passing alphas
        if you want to interpolate along that.

    Args:
        start (Union[torch.Tensor]): Starting point.
        end (Union[torch.Tensor]): Ending point.
        alpha (Union[torch.Tensor, List[float]]): Interpolation factor(s) (0.0 to 1.0).
    Returns:
        interpolated (Union[torch.Tensor, List[float]]): Interpolated point.
    """
    assert start.shape == end.shape, "Start and end must have the same shape."
    assert (alphas is not None) or (
        steps is not None
    ), "Either alpha or steps must be provided."

    if alphas is None:
        assert steps is not None, "Steps must be provided if alpha is None."
        alphas = torch.linspace(0, 1, steps).view((-1,) + (1,) * (start.ndim))

    alphas = alphas.to(start.device, dtype=start.dtype)

    return _linear_interpolate(
        start=start.unsqueeze(0), end=end.unsqueeze(0), alphas=alphas
    )


def _linear_interpolate(
    start: Union["np.ndarray", "torch.Tensor"],
    end: Union["np.ndarray", "torch.Tensor"],
    alphas: Optional[Union["np.ndarray", "torch.Tensor"]] = None,
) -> Union["np.ndarray", "torch.Tensor"]:

    return (1 - alphas) * start + alphas * end
def get_linear_projection_matrix_np(in_dims: int, out_dims: int):
    assert in_dims % out_dims == 0, "in_dims must be multiple of out_dims"
    matrix = np.zeros((out_dims, in_dims))
    factor = in_dims // out_dims
    for i in range(out_dims):
        matrix[i, i * factor : (i + 1) * factor] = 1.0 / factor
    return matrix
