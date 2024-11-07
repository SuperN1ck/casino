from typing import Optional
import logging


try:
    import open3d as o3d
except:
    logging.debug(
        "open3d not availble. Some functionality in visualization.py will break"
    )

try:
    import numpy as np
except:
    logging.debug(
        "numpy not availble. Some functionality in visualization.py will break"
    )


def get_o3d_coordinate_frame(
    transform: Optional["np.ndarray"] = None,
    position: Optional["np.ndarray"] = None,
    rotation: Optional["np.ndarray"] = None,
    scale: float = 1.0,
):
    """
    Returns an open3d coordinate mesh.

    `transform` superseeds `position` and `rotation`

    `transform` is a 4x4 numpy matrix TODO maybe also open3d?

    `position` is a 3x1 numpy vector/matrix, `rotation` is a 3x3 numpy matrix,
    """
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)

    if not transform is None:
        assert transform.shape == (4, 4)
        mesh.transform(transform)
        return mesh  # Early return!

    if not rotation is None:
        assert rotation.shape == (3, 3)
        mesh.rotate(rotation)  # Rotate around untranslated opbject --> origin

    if not position is None:
        assert position.shape[-1] == 3 and (position.ndim == 1 or position.ndim == 2)
        mesh.translate(position)

    return mesh
