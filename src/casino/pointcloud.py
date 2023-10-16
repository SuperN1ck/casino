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
    def from_matrix(matrix: np.ndarray):
        assert matrix.shape == (3, 3)
        return Intrinsics(
            f_x=matrix[0][0], f_y=matrix[1][1], c_x=matrix[0][2], c_y=matrix[1][2]
        )


def get_ordered(uvd: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
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


def get_xyz(depth: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
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
    depth: np.ndarray,
    intrinsics: Intrinsics,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    xyz = get_xyz(depth, intrinsics)

    if mask is None:
        mask = np.ones(depth.shape[:2], dtype=bool)
    else:
        assert depth.shape[:2] == mask.shape[:2]

    if mask.ndim == 3:
        mask = mask[..., 0]

    pc_noisy = xyz[mask]
    return pc_noisy[~np.any(np.isnan(pc_noisy), axis=-1)]
