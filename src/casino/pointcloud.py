import logging
from typing import Optional

try:
    import numpy as np
except:
    logging.debug("numpy not availble. Most functionality in pointcloud.py will break.")


try:
    import open3d as o3d
except:
    logging.debug(
        "open3d not available. Some functionality in pointcloud.py will break."
    )


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
    remove_out_of_bounds: bool = True,
) -> "np.ndarray":
    """
    Points should be in u-v format?
    u: vertical axis?
    v: horizontal axis?
    This potentially returns NaNs if depth is invalid/out of bounds
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
    u_clip = np.clip(u_crd[mask_uv], 0, depth.shape[0] - 1)
    v_clip = np.clip(v_crd[mask_uv], 0, depth.shape[1] - 1)

    pix_coords = np.stack(
        (
            v_clip,
            u_clip,
            (depth[..., 0] if depth.ndim == 3 else depth)[u_clip, v_clip],
        ),
        axis=-1,
    )
    _points_3d = get_ordered(pix_coords, intrinsics)

    if remove_out_of_bounds and rgb is None:
        return _points_3d
    elif remove_out_of_bounds and not rgb is None:
        return points_3d, rgb[u_clip, v_clip, :]

    # We do not remove bounds
    N_points = points.shape[0]
    points_3d = np.full((N_points, 3), np.nan)
    points_3d[mask_uv, :] = _points_3d
    if rgb is None:
        return points_3d

    colors = np.full((N_points, 3), np.nan)
    colors[mask_uv, :] = rgb[u_clip, v_clip]
    return points_3d, colors


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
    ).astype(depth.dtype)
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
        mask = mask.astype(bool)

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
    # TODO What about:
    # - flip x and y rows?
    # - flip y-axis because frame starts top-right?
    # No urgent ToDos, this seems to work, but it would be great to test this!
    return np.stack(
        (
            points[..., 0] / points[..., 2] * intrinsics.f_x + intrinsics.c_x,
            points[..., 1] / points[..., 2] * intrinsics.f_y + intrinsics.c_y,
        ),
        axis=-1,
    )


def transform_3d_points(
    points: "np.ndarray", transform: "np.ndarray", post_multiply: bool = False
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


def subsample(point_cloud: "np.ndarray", n_points: int, dim: int = 0):
    """
    Randomly subsample points from a point cloud
    """
    assert point_cloud.ndim == 2
    assert dim in [0, 1]

    idx = np.random.choice(point_cloud.shape[dim], n_points, replace=False)
    point_cloud = point_cloud.astype(np.float32)
    if dim == 0:
        return point_cloud[idx, :]
    elif dim == 1:
        return point_cloud[:, idx]


def to_o3d(
    pcd: "np.ndarray", color: "np.ndarray" = None, filter_invalid: bool = True
) -> "o3d.geometry.PointCloud":
    """
    Careful! if filter_nans == True, we will remove the Nan values in-place!
    """
    assert pcd.shape[1] == 3
    assert pcd.ndim == 2

    if filter_invalid:
        pcd = pcd[~np.isnan(pcd).any(axis=1), :]
        pcd = pcd[~np.isinf(pcd).any(axis=1), :]

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

    if not color is None:
        # TODO Add assert for correct scale?
        assert color.dtype in (np.float32, np.float64)
        assert color.min() >= 0.0 and color.max() <= 1.0
        if color.ndim == 1:
            color = np.expand_dims(color, 0)
        assert color.shape[1] == 3
        assert color.ndim == 2
        if color.shape[0] == 1:
            color = np.repeat(color, pcd.shape[0], axis=0)
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)

    return pcd_o3d
