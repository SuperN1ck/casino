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
        assert label(mask).max() == 1

    # Wrap AA BBOX around everything?
    if use_aa_bbox:
        new_coords = mask_to_coords(mask)
        min_h, min_w = new_coords.min(axis=0)
        max_h, max_w = new_coords.max(axis=0)
        mask[min_h:max_h, min_w:max_w] = True

    return mask
