from typing import Tuple
import logging

try:
    import numpy as np
except:
    logging.debug("numpy not availble. Most functionality in masks.py will break.")

try:
    import scipy
    from skimage.measure import label
except:
    logging.debug("scipy not availble. Most functionality in masks.py will break.")

try:
    import cv2
except:
    logging.debug("open-cv not availble. Most functionality in masks.py will break.")


try:
    from PIL import Image
except:
    logging.debug("PIL not availble. Some functionality in masks.py will break.")


def mask_to_coords(mask: "np.ndarray"):
    """
    Returns the pixel coordinates of a mask
    """
    assert mask.ndim == 2
    return np.array(np.where(mask)).T


def coords_to_mask(
    coords: "np.ndarray",
    mask_shape: Tuple,
    use_convex_hull: bool = False,
    use_dilation: bool = False,
    use_opening: bool = True,
    use_aa_bbox: bool = True,
):
    """
    Returns a mask
    """
    assert len(mask_shape) == 2
    assert coords.ndim == 2 and coords.shape[1] == 2

    mask = np.zeros(mask_shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    if use_convex_hull:
        # TODO https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
        # new_coords = mask_to_coords(mask)
        # hull_idx = scipy.spatial.ConvexHull(new_coords).vertices
        # hulled_mask = mask.copy().astype(np.uint8)
        # # negative thickness = filled
        # cv2.drawContours(hulled_mask, new_coords[hull_idx], -1, 1, thickness=-1)
        raise NotImplementedError()
    elif use_dilation:
        # TODO Maybe add an outlier filtering?
        dilation_iterations = 1
        while True and dilation_iterations < min(mask.shape) * 0.1:
            mask = scipy.ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            labels = label(mask)

            # Dilate by one pixel until there is only a single component left.
            if labels.max() == 1:
                break

            dilation_iterations += 1
    elif use_opening:
        mask = scipy.ndimage.binary_opening(mask)
        mask = scipy.ndimage.binary_fill_holes(mask)
        # assert label(mask).max() == 1

    # Wrap AA BBOX around everything?
    if use_aa_bbox:
        mask = make_mask_aa_bbox(mask)

    return mask


def filter_coords(points, shape, return_mask: bool = False):
    """
    Filters the given points to be within the given shape
    Mainly used if points i.e. coordinates could fall outside of an image
    """
    # Copied from Max :)
    u_crd, v_crd = points[:, 0], points[:, 1]
    # save positions that map to outside of bounds, so that they can be
    # set to 0
    mask_u = np.logical_or(u_crd < 0, u_crd >= shape[0])
    mask_v = np.logical_or(v_crd < 0, v_crd >= shape[1])
    mask_uv = np.logical_not(np.logical_or(mask_u, mask_v))

    # temporarily clip out of bounds values so that we can use numpy
    # indexing
    # TODO Why is this still here, should be filtered out anyway?
    # TODO Maybe make this an optional
    u_clip = np.clip(u_crd[mask_uv], 0, shape[0] - 1)
    v_clip = np.clip(v_crd[mask_uv], 0, shape[1] - 1)

    new_points = np.stack([u_clip, v_clip], axis=1)

    if return_mask:
        return new_points, mask_uv

    return new_points


def make_mask_aa_bbox(mask):
    """
    Inplace!
    """
    new_coords = mask_to_coords(mask)
    min_h, min_w = new_coords.min(axis=0)
    max_h, max_w = new_coords.max(axis=0)
    mask[min_h:max_h, min_w:max_w] = True
    return mask


def equal_max_bbox(masks: "np.ndarray", multiplier: float = 1.0):
    single_mask = masks.ndim == 2
    if single_mask:
        masks = masks.copy()[None, ...]

    assert masks.ndim == 3
    N_masks, H, W = masks.shape

    # coordinates: left, upper, right, lower
    all_bboxes = np.array([Image.fromarray(mask).getbbox() for mask in masks])
    all_widths = all_bboxes[:, 2] - all_bboxes[:, 0]
    all_heights = all_bboxes[:, 3] - all_bboxes[:, 1]
    all_centers = np.array(
        [all_bboxes[:, 0] + all_widths / 2, all_bboxes[:, 1] + all_heights / 2]
    ).T

    max_width = all_widths.max() * multiplier
    max_height = all_heights.max() * multiplier

    new_x_neg = all_centers[:, 0] - max_width / 2
    new_y_neg = all_centers[:, 1] - max_height / 2
    new_x_pos = all_centers[:, 0] + max_width / 2
    new_y_pos = all_centers[:, 1] + max_height / 2

    x_offset = 0
    x_offset = np.where(new_x_neg < 0.0, -new_x_neg, x_offset)
    x_offset = np.where(new_x_pos >= W, W - new_x_pos, x_offset)
    y_offset = 0
    y_offset = np.where(new_y_neg < 0.0, -new_y_neg, y_offset)
    y_offset = np.where(new_y_pos >= H, H - new_y_pos, y_offset)

    new_bboxes = np.ceil(
        np.array(
            [
                new_x_neg + x_offset,
                new_y_neg + y_offset,
                new_x_pos + x_offset,
                new_y_pos + y_offset,
            ]
        ).T
    ).astype(np.uint16)
    # TODO This needs logic for handling cases where the new bboundig box would be too big/small

    new_masks = np.zeros_like(masks, dtype=bool)
    for bbox_idx, new_bbox in enumerate(new_bboxes):
        new_masks[bbox_idx, new_bbox[1] : new_bbox[3], new_bbox[0] : new_bbox[2]] = True
    return new_masks if not single_mask else new_masks[0]


def mask_bbox(mask: "np.ndarray"):
    """
    Returns bbox in left, upper, right, lower format
    """
    new_coords = mask_to_coords(mask)
    min_h, min_w = new_coords.min(axis=0)
    max_h, max_w = new_coords.max(axis=0)
    return min_h, min_w, max_h, max_w


def mask_center_bbox(mask: "np.ndarray"):
    return np.median(mask_to_coords(mask), axis=0).astype(int)


def mask_center_geometric(mask: "np.ndarray"):
    return np.mean(mask_to_coords(mask), axis=0).astype(int)


def mask_top_left_bbox(mask: "np.ndarray"):
    return np.min(mask_to_coords(mask), axis=0).astype(int)


def get_segment_crop(img, mask):
    assert img.ndim == 2 or img.ndim == 3
    assert mask.ndim == 2
    return img[np.ix_(mask.any(1), mask.any(0))]
