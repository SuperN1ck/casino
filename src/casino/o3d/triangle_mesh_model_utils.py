from typing import List

import logging

try:
    import numpy as np
except:
    logging.debug("numpy not availble. line_mesh.py will break")

try:
    import open3d as o3d
except:
    logging.debug("open3d not availble. line_mesh.py will break")


def triangle_mesh_to_triangle_mesh_model(
    triangle_mesh: "o3d.geometry.TriangleMesh", transform_to_center: bool = True
) -> "o3d.visualization.rendering.TriangleMeshModel":
    """
    Be careful, all operations are performed in-place
    """
    default_material = o3d.visualization.rendering.MaterialRecord()
    default_material.base_color = [1.0, 1.0, 1.0, 1.0]
    default_material.shader = "defaultLit"
    default_material.point_size = 5
    if len(triangle_mesh.textures) == 1:
        default_material.albedo_img = o3d.geometry.Image(triangle_mesh.textures[0])

    if transform_to_center:
        triangle_mesh.translate(-triangle_mesh.get_center())

    triangle_mesh_model = o3d.visualization.rendering.TriangleMeshModel()
    triangle_mesh_model.meshes = [
        triangle_mesh_model.MeshInfo(
            triangle_mesh, "mesh_0", 0  # mesh name  # material index
        )
    ]
    triangle_mesh_model.materials = [default_material]
    return triangle_mesh_model


def get_extent(mesh_model: "o3d.visualization.rendering.TriangleMeshModel"):
    extent = np.array(
        [
            [np.inf, np.inf, np.inf],
            [-np.inf, -np.inf, -np.inf],
        ]
    )
    for mesh_info in mesh_model.meshes:
        local_mesh = mesh_info.mesh
        local_bbox = local_mesh.get_axis_aligned_bounding_box()
        extent[0] = np.minimum(extent[0], local_bbox.min_bound)
        extent[1] = np.maximum(extent[1], local_bbox.max_bound)
    return extent


def get_center(mesh_model: "o3d.visualization.rendering.TriangleMeshModel"):
    return get_extent(mesh_model).mean(axis=0)


# TODO [NH]: 2025/09/05: Define method to apply affine transformation on all meshes


def translate(
    mesh_model: "o3d.visualization.rendering.TriangleMeshModel",
    translation: List[float] = [0.0, 0.0, 0.0],
):
    translation = np.array(translation)
    for mesh_info in mesh_model.meshes:
        local_mesh: o3d.geometry.TriangleMesh = mesh_info.mesh
        local_mesh.translate(translation)


def scale(
    mesh_model: "o3d.visualization.rendering.TriangleMeshModel",
    scale: float = 1.0,
    center: List[float] = [
        0.0,
        0.0,
        0.0,
    ],
):
    center = np.array(center)
    for mesh_info in mesh_model.meshes:
        local_mesh: o3d.geometry.TriangleMesh = mesh_info.mesh
        local_mesh.scale(scale, center)
