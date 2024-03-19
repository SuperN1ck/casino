from typing import Tuple, List, Optional, Union
import logging
import copy
from .line_mesh import LineMesh

try:
    import open3d as o3d
except:
    logging.debug(
        "open3d not availble. Some functionality in pointcloud_offline_renderer.py will break"
    )


try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except:
    logging.debug(
        "matplotlib not availble. Some functionality in pointcloud_offline_renderer.py will break"
    )

try:
    import numpy as np
except:
    logging.debug(
        "numpy not availble. Some functionality in pointcloud_offline_renderer.py will break"
    )


try:
    from scipy.spatial import transform
except:
    logging.debug(
        "scipy not availble. Some functionality in pointcloud_offline_renderer.py will break"
    )


from ..visualization import get_o3d_coordinate_frame


def get_o3d_render(
    frame_width: int = 1200,
    frame_height: int = 1200,
    flip_viewing_direction: bool = False,
):
    renderer = o3d.visualization.rendering.OffscreenRenderer(frame_width, frame_height)
    vertical_field_of_view = 60.0  # between 5 and 90 degrees
    aspect_ratio = frame_width / frame_height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 10
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    renderer.scene.camera.set_projection(
        vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type
    )

    center = [0.0, 0.0, 0.0]  # look_at target
    eye = [-1.0 if not flip_viewing_direction else 1.0, -1.0, 1.0]
    up = [0.0, 0.0, 1.0]  # camera orientation

    renderer.scene.camera.look_at(center, eye, up)
    # renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Black?
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # White?
    renderer.scene.set_lighting(
        o3d.visualization.rendering.Open3DScene.NO_SHADOWS,
        (0.0, 0.0, -1.0),  # (0.577, -0.577, -0.577)
    )

    return renderer


def render_o3d_mesh(
    all_pcd: Tuple[o3d.geometry.PointCloud, List[o3d.geometry.PointCloud]],
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    height_coloring=False,
    frame_width=1200,
    frame_height=1200,
    flip_viewing_direction: bool = True,
):
    renderer.scene.clear_geometry()
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    # renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    # material = o3d.visualization.rendering.Material()
    material = o3d.visualization.rendering.MaterialRecord()
    # material.base_color = [1.0, 0.75, 0.0, 1.0]
    # material.base_color = [((1 / 255) * 223), ((1 / 255) * 116), ((1 / 255) * 10), 1.0]
    # material.base_color = [0.0, 0.0, 0.0, 1.0]
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.point_size = 5.0  # Default is 3
    # print(f"{material.base_color = }")
    # material.base_color = o3d.utility.Vector3dVector(np.array(in_pcds[0].normals))
    # material.shader = "litPoints"

    # constant_color = np.ones((np.array(pcd.points).shape[0], 3)) * np.array([1.0, 0.75, 0.0])
    # pc_color = constant_color

    if not isinstance(all_pcd, list):
        all_pcd = [all_pcd]

    for idx, pcd in enumerate(all_pcd):
        if height_coloring and isinstance(pcd, o3d.geometry.PointCloud):
            points = np.array(pcd.points)
            # assign colors to the pointcloud file
            cmap_norm = mpl.colors.Normalize(
                vmin=points[:, 2].min(), vmax=points[:, 2].max()
            )
            #'hsv' is changeable to any name as stated here: https://matplotlib.org/stable/tutorials/colors/colormaps.html
            pc_color = plt.get_cmap("jet")(cmap_norm(np.asarray(points)[:, 2]))[:, 0:3]

            pcd.estimate_normals()  # TODO Move out maybe?
            pcd.colors = o3d.utility.Vector3dVector(pc_color)

        renderer.scene.add_geometry(f"pcd_{idx}", pcd, material)

    # Coordinate Frame
    # renderer.scene.add_geometry(
    #     "coordinate_frame",
    #     o3d.geometry.TriangleMesh.create_coordinate_frame(
    #         size=0.3, origin=[-0.51307064, -0.11901558, 2.38963232]
    #     ), material
    # )

    img = renderer.render_to_image()
    np_image = np.asarray(img)  # [::2, ::2, :]
    # images.append(Image.fromarray((np_image * 255).astype(np.uint8)))
    return np_image


def render_rotate_around_o3d_meshes(
    all_pcds: Tuple[o3d.geometry.PointCloud, List[o3d.geometry.PointCloud]],
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    fps: float = 10,
    duration: float = 2,
    rotation_center: Optional[Union[List[float], "np.ndarray"]] = None,
    rotation_axis: Optional[Union[List[float], "np.ndarray"]] = [0.0, 0.0, 1.0],
    camera_position: Optional[Union[List[float], "np.ndarray"]] = None,
    up_vector: Optional[Union[List[float], "np.ndarray"]] = [0.0, -1.0, 0.0],
    radius=2,
    debug_vis: bool = False,
):
    """
    Will rotate around the given pointclouds,
    `rotation_center`: is not given, we will take the center of all input pointclouds
    `rotation_axis`: if not given we rotate around z
    `camera_position` we will move back in all direction rotaiton center by 1/3 of the max size of the input pointclouds

    """
    all_pcd = copy.deepcopy(all_pcds)

    # TODO Figure out a way to save the camera?
    # input_camera = copy.deepcopy(renderer.scene.camera)

    if rotation_center is None or camera_position is None:
        all_maxs = np.array([o3d_object.get_max_bound() for o3d_object in all_pcd])
        all_mins = np.array([o3d_object.get_min_bound() for o3d_object in all_pcd])
        max_corner = all_maxs.max(axis=0)
        min_corner = all_mins.min(axis=0)

    if rotation_center is None:
        rotation_center = min_corner + (max_corner - min_corner) / 2
    rotation_center = np.asarray(rotation_center)

    if camera_position is None:
        bbox_size = np.linalg.norm(max_corner - min_corner)
        camera_position = (rotation_center - 1 / 3 * bbox_size).copy()
    camera_position = np.asarray(camera_position)
    camera_position_homogenous = np.append(camera_position, 1)

    rot_to_cam = camera_position - rotation_center
    rot_to_cam_homogenous = np.append(rot_to_cam, 1)

    if debug_vis:
        all_pcd.append(get_o3d_coordinate_frame(position=rotation_center, scale=0.15))
        all_pcd.extend(
            LineMesh(
                points=[
                    rotation_center - rotation_axis,
                    rotation_center + rotation_axis,
                ],
                lines=[[0, 1]],
                radius=0.05,
            ).get_segments()
        )
        all_pcd.append(get_o3d_coordinate_frame(position=camera_position, scale=0.15))

    all_images = []
    for frame_i, rotation in enumerate(
        np.linspace(0, 2 * np.pi, endpoint=False, num=fps * duration)
    ):
        camera_rotation_R = transform.Rotation.from_rotvec(
            rotation * np.asarray(rotation_axis)
        )
        camera_R = np.eye(4)
        camera_R[:3, :3] = camera_rotation_R.as_matrix()

        camera_position_i = (camera_R @ rot_to_cam_homogenous)[
            :3
        ] - camera_position  # Extract only relevant part from homogeneous coordinate

        # up_vector_i = up_vector
        up_vector_i = camera_rotation_R.as_matrix() @ up_vector

        # center: where we look at
        # eye: position in world
        # up: up vector for camera image (0, -1, 0) --> look in z-direction
        renderer.scene.camera.look_at(rotation_center, camera_position_i, up_vector_i)

        image = render_o3d_mesh(all_pcd, renderer=renderer)
        all_images.append(image)

        # Append debug stuff
        if debug_vis:
            all_pcd.append(
                get_o3d_coordinate_frame(position=camera_position_i, scale=0.1)
            )

    # and then reapply..
    # renderer.scene.camera = input_camera

    if debug_vis:
        o3d.visualization.draw_geometries(all_pcd)

    return all_images
