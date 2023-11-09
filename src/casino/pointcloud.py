import logging
from typing import Optional

try:
    import numpy as np
except:
    logging.debug("numpy not availble. Most functionality in pointcloud.py will break")


class Intrinsics:
    def __init__(
        self: "Intrinsics",
        # these are not actually needed --> can we somehow ensure that our intrinsics is on pixel level?
        # height: float,
        # width: float,
        f_x: float,
        f_y: float,
        c_x: float,
        c_y: float,
    ):
        # self.height = height
        # self.width = width
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y

    @property
    def matrix(self: "Intrinsics"):
        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = self.f_x
        intrinsics[1, 1] = self.f_y
        intrinsics[0, 2] = self.c_x
        intrinsics[1, 2] = self.c_y
        intrinsics[2, 2] = 1.0
        return intrinsics

    @staticmethod
    def from_matrix(matrix: "np.ndarray"):
        assert matrix.shape == (3, 3)
        return Intrinsics(
            f_x=matrix[0][0], f_y=matrix[1][1], c_x=matrix[0][2], c_y=matrix[1][2]
        )


def get_ordered(uvd: "np.ndarray", intrinsics: Intrinsics) -> "np.ndarray":
    """
    uvd is any tensor where the last dimension should have dimension 3, with following content:
        u: u coordinate in the image
        v: v coordinate in the image
        d: corresponding measured, depth value in meters
    """
    assert uvd.shape[-1] == 3

    xyz_noisy = np.stack(
        (
            (uvd[..., 0] - intrinsics.c_x) * uvd[..., 2] / intrinsics.f_x,
            (uvd[..., 1] - intrinsics.c_y) * uvd[..., 2] / intrinsics.f_y,
            uvd[..., 2],
        ),
        axis=-1,
    )
    xyz = np.where(uvd[..., 2][..., None] > 0.0, xyz_noisy, np.nan)
    return xyz


def get_points(
    points: "np.ndarray",
    depth: "np.ndarray",
    intrinsics: Intrinsics,
    rgb: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """
    Points should be in u-v format?
    u: vertical axis?
    v: horizontal axis?
    """
    assert points.shape[1] == 2 and points.ndim == 2

    # Copied from Max :)
    u_crd, v_crd = points[:, 0], points[:, 1]
    # save positions that map to outside of bounds, so that they can be
    # set to 0
    mask_u = np.logical_or(u_crd < 0, u_crd >= depth.shape[0])
    mask_v = np.logical_or(v_crd < 0, v_crd >= depth.shape[1])
    mask_uv = np.logical_not(np.logical_or(mask_u, mask_v))
    # temporarily clip out of bounds values so that we can use numpy
    # indexing
    u_clip = np.clip(u_crd, 0, depth.shape[0] - 1)
    v_clip = np.clip(v_crd, 0, depth.shape[1] - 1)

    logging.debug("Found out-of-bounds values for ")

    pix_coords = np.stack(
        (
            v_clip,
            u_clip,
            (depth[..., 0] if depth.ndim == 3 else depth)[u_clip, v_clip],
        ),
        axis=-1,
    )
    points_3d = get_ordered(pix_coords, intrinsics)
    if rgb is None:
        return points_3d
    return points_3d, rgb[u_clip, v_clip, :]


def get_xyz(depth: "np.ndarray", intrinsics: Intrinsics) -> "np.ndarray":
    """
    Returns xyz image,
    depth <= 0.0 will be nan
    """
    u, v = np.meshgrid(
        np.arange(depth.shape[1]),
        np.arange(depth.shape[0]),
    )

    pix_coords = np.stack(
        (
            u,
            v,
            (depth[..., 0] if depth.ndim == 3 else depth),
        ),
        axis=-1,
    )
    return get_ordered(pix_coords, intrinsics)


def get_pc(
    depth: "np.ndarray",
    intrinsics: Intrinsics,
    mask: Optional["np.ndarray"] = None,
    rgb: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    xyz = get_xyz(depth, intrinsics)

    if mask is None:
        mask = np.ones(depth.shape[:2], dtype=bool)
    else:
        assert depth.shape[:2] == mask.shape[:2]

    if mask.ndim == 3:
        mask = mask[..., 0]

    pc_noisy = xyz[mask]
    valid_indices = ~np.any(np.isnan(pc_noisy), axis=-1)

    pc_clean = pc_noisy[valid_indices]
    if rgb is None:
        return pc_clean

    return pc_clean, rgb.copy()[mask][valid_indices]


# TODO Add optional argument specifying to ensure correct numbers
def make_homogeneous(points: "np.ndarray") -> "np.ndarray":
    """
    Adds a one at the end of the second dimension
    """
    assert points.ndim == 2
    return np.hstack((points, np.ones((points.shape[0], 1))))


def make_non_homoegeneous(points: "np.ndarray") -> "np.ndarray":
    assert points.ndim == 2
    """
    Divides last dimensions with the last entry
    """
    return points[:, :-1] / points[:, -1][..., None]


def project_onto_image(points: "np.ndarray", intrinsics: Intrinsics) -> "np.ndarray":
    assert points.shape[-1] == 3
    return np.stack(
        (
            points[..., 0] / points[..., 2] * intrinsics.f_x + intrinsics.c_x,
            points[..., 1] / points[..., 2] * intrinsics.f_y + intrinsics.c_y,
        ),
        axis=-1,
    )


def transform_3d_points(
    points: "np.ndarray", transform: np.ndarray, post_multiply: bool = False
) -> "np.ndarray":
    """
    Post multiplies transform
    """
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert transform.shape[-2:] == (4, 4)

    points_hom = make_homogeneous(points)
    if post_multiply:
        points_trans = points_hom @ transform
    else:  # We do pre-multiplying :)
        points_trans = (transform @ points_hom.T).T

    return make_non_homoegeneous(points_trans)
