import logging
from typing import Optional, Union

from .masks import filter_coords
from .special_dicts import deindex

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


try:
    import torch
except:
    logging.debug(
        "torch not available. Some functionality in pointcloud.py will break."
    )


def degree_to_focal_length(degrees):
    return (1.0 / np.tan(np.deg2rad(degrees) / 2)) / 2.0


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


def get_ordered(
    uvd: "np.ndarray", intrinsics: Union[Intrinsics, "np.ndarray"]
) -> "np.ndarray":
    """
    uvd is any tensor where the last dimension should have dimension 3, with following content:
        u: u coordinate in the image
        v: v coordinate in the image
        d: corresponding measured, depth value in meters
    """
    assert uvd.shape[-1] == 3

    if isinstance(intrinsics, np.ndarray):
        intrinsics = Intrinsics.from_matrix(intrinsics)

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
    intrinsics: Union[Intrinsics, "np.ndarray"],
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
    if depth.ndim == 3:
        depth = depth[..., 0]

    filtered_points, mask_uv = filter_coords(points, depth.shape, return_mask=True)
    u_clip, v_clip = filtered_points[..., 0], filtered_points[..., 1]

    pix_coords = np.stack(
        (
            v_clip,
            u_clip,
            depth[u_clip, v_clip],
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


def get_xyz(
    depth: "np.ndarray", intrinsics: Union[Intrinsics, "np.ndarray"]
) -> "np.ndarray":
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
    intrinsics: Union[Intrinsics, "np.ndarray"],
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
    assert points.shape[-1] == 4
    """
    Divides last dimensions with the last entry
    """
    return points[..., :-1] / points[..., -1][..., None]


def make_homogeneous_th(points: "torch.Tensor") -> "torch.Tensor":
    """
    Adds a one at the end of the last dimension
    """
    assert points.shape[-1] == 3
    return torch.cat(
        (
            points,
            torch.ones(
                (*points.shape[:-1], 1),
                device=points.device,
            ),
        ),
        dim=-1,
    )


def make_non_homoegeneous_th(points: "torch.Tensor") -> "torch.Tensor":
    """
    Divides last dimensions with the last entry
    """
    return make_non_homoegeneous(points)


def project_onto_image(
    points: "np.ndarray", intrinsics: Union[Intrinsics, "np.ndarray"]
) -> "np.ndarray":
    assert points.shape[-1] == 3

    if isinstance(intrinsics, np.ndarray):
        intrinsics = Intrinsics.from_matrix(intrinsics)

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


def transform_3d_points_th(
    points: "torch.Tensor", transform: "torch.Tensor", post_multiply: bool = False
) -> "torch.Tensor":
    assert points.shape[-1]
    B_points = points.shape[:-2]
    B_transform = transform.shape[:-2]

    if len(B_transform) == 0 and len(B_points) > 0:
        transform = transform.view((1,) * len(B_points) + (4, 4))
        transform = transform.repeat(*B_points, 1, 1)
    elif len(B_transform) == len(B_points):
        repeats = [
            b_points // b_transform
            for b_points, b_transform in zip(B_points, B_transform)
        ]
        transform = transform.repeat(*repeats, 1, 1)
    else:
        assert B_transform == B_points

    points_hom = make_homogeneous_th(points)
    if post_multiply:
        points_trans = torch.einsum("...nj,...ij->...ni", points_hom, transform)
    else:
        points_trans = torch.einsum("...ij,...nj->...ni", transform, points_hom)

    return make_non_homoegeneous_th(points_trans)


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


def subsample_th(pointcloud: "torch.Tensor", n_points: int, sample_dim: int = -2):
    """
    Uniformly subsamples a pointcloud to `n_points` along the `sample_dim`.
    Indices are shuffled for each batch entry.

    Assumes the shape of
        N x 3
    or
        3 x N
    or
        B_1 x ... x B_b x N x 3
    or
        B_1 x ... x B_b x 3 x N

    sample_dim defines the dimesions along which we sample
        - should be -1 or -2
        - default (sample_dim=-2) is assuming N x 3
    """
    og_tensor_shape = list(pointcloud.size())

    # Flat away all batch dimensions
    flat_pc = (
        pointcloud.flatten(0, -3)
        if len(og_tensor_shape) > 2
        else pointcloud.unsqueeze(0)
    )

    B = flat_pc.size(0)
    in_n_points = flat_pc.size(sample_dim)

    # For each entry create
    indices = [torch.randperm(in_n_points)[:n_points] for _ in range(B)]
    stacked_indices = torch.stack(indices, dim=0).to(flat_pc.device)

    if sample_dim in [-2, 1]:
        stacked_indices = stacked_indices.unsqueeze(-1).expand(-1, -1, 3)
    elif sample_dim in [-1, 2]:
        stacked_indices = stacked_indices.unsqueeze(-2).expand(-1, 3, -1)
    else:
        raise NotImplementedError(f"Unknown {sample_dim = }")

    sampled_pc = torch.gather(flat_pc, dim=sample_dim, index=stacked_indices)

    # Overwrite original point amount with new size
    og_tensor_shape[sample_dim] = n_points
    og_size_sampled_pc = sampled_pc.reshape(og_tensor_shape)
    return og_size_sampled_pc


def crop_mask_aabb(point_cloud: "np.ndarray", aabb: "np.ndarray"):
    """
    Extracts the mask for points in an axis-aligned bounding box
    We assume the bounding box is given in the format of
    [[-x, x],
     [-y, y],
     [-z, z]]

    returns a view of the original points
    """
    # Logical conditions for each axis
    mask = (
        (point_cloud[..., 0] >= aabb[0, 0])
        & (point_cloud[..., 0] <= aabb[0, 1])  # X-axis
        & (point_cloud[..., 1] >= aabb[1, 0])
        & (point_cloud[..., 1] <= aabb[1, 1])  # Y-axis
        & (point_cloud[..., 2] >= aabb[2, 0])
        & (point_cloud[..., 2] <= aabb[2, 1])  # Z-axis
    )
    return mask


def crop_aabb(point_cloud: "np.ndarray", aabb: "np.ndarray", in_place: bool = True):
    mask = crop_mask_aabb(point_cloud, aabb)
    if not in_place:
        point_cloud = point_cloud.copy()
    # Apply the mask to filter points
    return point_cloud[mask]


# Perfoms a weighted distance measure to the points
def weighted_distance_to_point_th(
    points: "torch.Tensor",
    anchor_points: "torch.Tensor",
    data_dim: int = -1,  # e.g. should be 3
    points_dim: int = -2,  # dimension for point
    scale: float = 1,
):
    """
    Bump up scale to make the curve sharper, i.e. softmax values more distinct
    """
    # TODO Make it batched? Is this needed?
    # i.e. what happens if anchor points has size
    #   3
    # or
    #   N_points x 3
    # does this still work?

    # if points.ndim == 2:
    #     points = points.unsqueeze(0)
    # assert points.shape[points_dim] == 3

    if anchor_points.ndim > 1:
        # Make sure they share the same dimensions
        assert deindex(anchor_points.shape, data_dim) == deindex(points.shape, data_dim)

    dist = torch.norm(points - anchor_points, dim=data_dim, keepdim=True)
    weights = torch.softmax((1 / dist + 1e-7) / scale, dim=points_dim)

    # fig, axes = plt.subplots(ncols=2, nrows=1)
    # axes[0].hist(dist[0, :, 0])
    # axes[0].set_xlim(0.0, 3.0)  # from 0 to 3m euclid
    # axes[1].hist(weights[0, :, 0])
    # # axes[1].set_xlim(0.0, 0.001)
    # plt.show()
    # print(f"{weights.min() = }")
    # print(f"{weights.max() = }")

    return weights, dist


def to_o3d(
    pcd: "np.ndarray", color: "np.ndarray" = None, filter_invalid: bool = True
) -> "o3d.geometry.PointCloud":
    """
    Careful! if filter_nans == True, we will remove the Nan values in-place!
    """
    assert pcd.shape[1] == 3
    assert pcd.ndim == 2

    if filter_invalid:
        invalid_filter = ~np.isnan(pcd).any(axis=1) & ~np.isinf(pcd).any(axis=1)
        pcd = pcd[invalid_filter, :]

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
        elif filter_invalid:
            color = color[invalid_filter, :]
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)

    return pcd_o3d
