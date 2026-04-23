# Adapted from here: https://github.com/robot-learning-freiburg/PointFlowMatch/blob/main/pfp/common/visualization.py

import logging
from typing import Optional, Set

# from yourdfpy.urdf import URDF

try:
    import rerun as rr

    assert int(rr.__version__.split(".")[0]) >= 0
    assert int(rr.__version__.split(".")[1]) >= 25
except:
    logging.debug(
        "rerun-sdk>=0.25 not availble. Most functionality in rerun_wrapper.py will break."
    )

try:
    import numpy as np
except:
    logging.debug(
        "numpy not availble. Some functionality in rerun_wrapper.py will break."
    )

try:
    import open3d as o3d
except:
    logging.debug(
        "open3d not availble. Some functionality in rerun_wrapper.py will break."
    )

try:
    import trimesh
except:
    logging.debug(
        "trimesh not availble. Some functionality in rerun_wrapper.py will break."
    )


class RerunViewer:
    entity_names: Set[str] = set()
    # This is the core of rerun and will hold all the data.
    recording: Optional["rr.RecordingStream"] = None

    # Add?
    # __init__(self):
    #     ...
    # and
    # __init__(self, ...):
    #    self.init_viewer(...)

    def add_name_to_entity_list(func):
        def wrapper_func(self, *args, **kwargs):
            # Only allow logging when we have an instance running here
            if self.recording is None:
                logging.warning("DefaultRerunViewer was not initialized.")
                return

            # The name of the added entity is always the first argument(!)
            self.entity_names.add(args[0])

            # Then log as normal
            func(self, *args, **kwargs)

        return wrapper_func

    def init_viewer(
        self, name: str, addr: str = None, recording_id: Optional[str] = None
    ):
        if self.recording is None:
            self.recording = rr.RecordingStream(name, recording_id=recording_id)
            self.recording.spawn()

        if addr is None:
            addr = "127.0.0.1"
        port = "9876"
        uri = f"rerun+http://{addr}:{port}/proxy"

        rr.connect_grpc(uri, recording=self.recording)

    def clear(self, recursive: bool = True):
        for entity_name in self.entity_names:
            self.recording.log(entity_name, rr.Clear(recursive=recursive))
        # self.entity_names.clear()

    def set_time_sequence(self, timeline_name, time_index):
        rr.set_time(
            timeline=timeline_name, sequence=time_index, recording=self.recording
        )

    # TODO Re-activate? RerunViewer --> self
    # @staticmethod
    # def add_obs_dict(obs_dict: dict, timestep: int = None):
    #     if timestep is not None:
    #         rr.set_time_sequence("timestep", timestep)
    #     RerunViewer.add_rgb("rgb", obs_dict["image"])
    #     RerunViewer.add_depth("depth", obs_dict["depth"])
    #     RerunViewer.add_np_pointcloud(
    #         "vis/pointcloud",
    #         points=obs_dict["point_cloud"][:, :3],
    #         colors_uint8=obs_dict["point_cloud"][:, 3:],
    #     )

    @add_name_to_entity_list
    def add_o3d_pointcloud(
        self, name: str, pointcloud: "o3d.geometry.PointCloud", radii: float = None
    ):
        points = np.asanyarray(pointcloud.points)
        colors = np.asanyarray(pointcloud.colors) if pointcloud.has_colors() else None
        colors_uint8 = (
            (colors * 255).astype(np.uint8) if pointcloud.has_colors() else None
        )
        self.add_np_pointcloud(name, points, colors_uint8, radii)

    @add_name_to_entity_list
    def add_np_pointcloud(
        self,
        name: str,
        points: "np.ndarray",
        colors_uint8: "np.ndarray" = None,
        normals_float32: "np.ndarray" = None,
        normal_length: float = 0.1,
        radii: float = None,
    ):
        rr_points = rr.Points3D(positions=points, colors=colors_uint8, radii=radii)
        self.recording.log(name, rr_points)

        if normals_float32 is not None:
            self.recording.log(
                f"{name}/normals",
                rr.Arrows3D(origins=points, vectors=normals_float32 * normal_length),
            )

    @add_name_to_entity_list
    def add_axis(
        self,
        name: str,
        pose: "np.ndarray",
        size: float = 0.01,
        origin_color=None,
        static: bool = False,
    ):
        mesh = trimesh.creation.axis(
            origin_size=size, origin_color=origin_color, transform=pose
        )
        self.add_mesh_trimesh(name, mesh, static=static)

    add_pose = add_axis

    @add_name_to_entity_list
    def add_aabb(
        self,
        name: str,
        centers: Optional["np.ndarray"] = None,
        extents: Optional["np.ndarray"] = None,
        corners: Optional["np.ndarray"] = None,
        static: bool = False,
    ):
        """
        Adds a bounding box, if we provide corners, we will wrote centers + extents
        """
        if not corners is None:
            centers = corners.mean(axis=-1)
            extents = corners[:, 1] - corners[:, 0]

        self.recording.log(
            name, rr.Boxes3D(centers=centers, sizes=extents), static=static
        )

    @add_name_to_entity_list
    def add_mesh_trimesh(
        self, name: str, mesh: "trimesh.Trimesh", static: bool = False
    ):
        # Handle colors
        if mesh.visual.kind in ["vertex", "face"]:
            vertex_colors = mesh.visual.vertex_colors
        elif mesh.visual.kind == "texture":
            vertex_colors = mesh.visual.to_color().vertex_colors
        else:
            vertex_colors = None
        # Log mesh
        rr_mesh = rr.Mesh3D(
            vertex_positions=mesh.vertices,
            vertex_colors=vertex_colors,
            vertex_normals=mesh.vertex_normals,
            triangle_indices=mesh.faces,
        )
        self.recording.log(name, rr_mesh, static=static)

    @add_name_to_entity_list
    def add_mesh_list_trimesh(self, name: str, meshes: list["trimesh.Trimesh"]):
        for i, mesh in enumerate(meshes):
            self.add_mesh_trimesh(name + f"/{i}", mesh)

    @add_name_to_entity_list
    def add_rgb(self, name: str, rgb_uint8: "np.ndarray"):
        if rgb_uint8.shape[0] == 3:
            # CHW -> HWC
            rgb_uint8 = np.transpose(rgb_uint8, (1, 2, 0))
        self.recording.log(name, rr.Image(rgb_uint8))

    @add_name_to_entity_list
    def add_depth(self, name: str, depth: "np.ndarray", **kwargs):
        self.recording.log(name, rr.DepthImage(depth), **kwargs)

    # @add_name_to_entity_list
    # Is done automatically since we call add_axis
    def add_world_coordinate_system(self, name: str = "world", size: float = 0.05):
        self.add_axis(name, np.eye(4), size=size, static=True)

    @add_name_to_entity_list
    def add_scalar(self, name: str, scalar_value):
        self.recording.log(name, rr.Scalars(np.array(scalar_value)))

    def add_array(self, name: str, array: "np.ndarray", **kwargs):
        self.recording.log(name, rr.Tensor(array, **kwargs))

    @add_name_to_entity_list
    def add_text(self, name: str, text: str, **kwargs):
        self.recording.log(name, rr.TextLog(text, **kwargs))

    @add_name_to_entity_list
    def add_scatter(
        self, name: str, points_2D: "np.ndarray", point_kwargs={}, log_kwargs={}
    ):
        self.recording.log(
            name, rr.Points2D(positions=points_2D, **point_kwargs), **log_kwargs
        )

    @add_name_to_entity_list
    def add_hist(self, name: str, data: "np.ndarray", normalize=False, hist_kwargs={}, log_kwargs={}):
        """
        Pass arguments to np.histogram via hist_kwargs.
        """
        assert data.ndim == 1, "Data must be 1D."
        counts, bins = np.histogram(data, **hist_kwargs)
        if normalize:
            counts = counts / counts.sum()
        self.recording.log(
            name, rr.BarChart(values=counts, abscissa=bins), **log_kwargs
        )

    @add_name_to_entity_list
    def add_np_voxel_grid(
        self,
        name: str,
        occupancy: "np.ndarray",
        point_size_multiplier: float = 0.5,
        grid_dimensions: Optional["np.ndarray"] = np.array(
            [
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
            ]
        ),
        static: bool = False,
        color_uint8=None,
        normals_float32=None,
    ):
        """
        voxel_grid: 3D numpy array of bools or ints (1 = occupied, 0 = free)
        voxel_size: size of each voxel
        origin: origin of the voxel grid in world coordinates
        """
        assert occupancy.ndim == 3, "Occupancy grid must be 3D."
        assert grid_dimensions.shape == (3, 2), "Grid dimensions must be (3, 2)."

        occupied_indices = np.argwhere(occupancy)
        if occupied_indices.shape[0] == 0:
            logging.warning(f"No occupied voxels to add for {name}.")
            return

        grid_min = grid_dimensions[:, 0]
        grid_max = grid_dimensions[:, 1]
        grid_size = grid_max - grid_min
        voxel_size = grid_size / np.array(occupancy.shape)
        points = grid_min + occupied_indices * voxel_size + voxel_size / 2.0
        point_colors = (
            color_uint8.reshape(np.prod(color_uint8.shape[:3]), 3)[
                np.ravel_multi_index(occupied_indices.T, color_uint8.shape[:3])
            ]
            if color_uint8 is not None
            else None
        )
        point_normals = (
            
            normals_float32.reshape(np.prod(normals_float32.shape[:3]), 3)[
                np.ravel_multi_index(occupied_indices.T, normals_float32.shape[:3])
            ]
            if normals_float32 is not None
            else None
        )
        # self.recording.log(
        #     name,
        #     rr.Points3D(positions=points, radii=voxel_size / 2.0),
        #     static=static,
        # )
        # self.recording.log(name, rr.Tensor(voxels, dim_names=["right", "back", "up"]))
        self.add_np_pointcloud(
            name,
            points,
            colors_uint8=point_colors,
            normals_float32=point_normals,
            radii=voxel_size.max() * point_size_multiplier,
            normal_length=(grid_max - grid_min).mean() / 20.0,
        )

    @add_name_to_entity_list
    def add_tensor(
        self,
        name,
        tensor,
        tensor_kwargs={},
        log_kwargs={},
    ):
        self.recording.log(name, rr.Tensor(tensor, **tensor_kwargs), **log_kwargs)

    @add_name_to_entity_list
    def add_axis_rr(self, name, pose, size=0.01, origin_color=None, static=False):
        self.recording.log(
            name,
            rr.Transform3D.from_fields(
                mat3x3=pose[:3, :3],
                translation=np.array(pose[:3, 3]),
            ),
            rr.TransformAxes3D(size),
            static=static,
        )

    # @staticmethod
    # def add_pose_trajectory():
    # f"{name}/{i}t"

    # @staticmethod
    # def add_traj(name: str, traj: "np.ndarray"):
    #     """
    #     name: str
    #     traj: np.ndarray (T, 10)
    #     """
    #     poses = pfp_to_pose_np(traj)
    #     for i, pose in enumerate(poses):
    #         RerunViewer.add_axis(name + f"/{i}t", pose)
    #     return


try:
    DefaultRerunViewer = RerunViewer()
except:
    logging.debug("Could not initialize default RerunViewer.")


# TODO Re-Enable?
# class RerunTraj:
#     def __init__(self) -> None:
#         self.traj_shape = None
#         return

#     def add_traj(self, name: str, traj: "np.ndarray", size: float = 0.004):
#         """
#         name: str
#         traj: np.ndarray (T, 10)
#         """
#         if self.traj_shape is None or self.traj_shape != traj.shape:
#             self.traj_shape = traj.shape
#             for i in range(traj.shape[0]):
#                 RerunViewer.add_axis(name + f"/{i}t", np.eye(4), size)
#         poses = pfp_to_pose_np(traj)
#         for i, pose in enumerate(poses):
#             self.recording.log(
#                 name + f"/{i}t",
#                 rr.Transform3D(mat3x3=pose[:3, :3], translation=pose[:3, 3]),
#             )
#         return


# class RerunURDF:
#     def __init__(self, name: str, urdf_path: str, meshes_root: str):
#         self.name = name
#         self.urdf: URDF = URDF.load(urdf_path, mesh_dir=meshes_root)
#         return

#     def update_vis(
#         self,
#         joint_state: list | np.ndarray,
#         root_pose: np.ndarray = np.eye(4),
#         name_suffix: str = "",
#     ):
#         self._update_joints(joint_state)
#         scene = self.urdf.scene
#         trimeshes = self._scene_to_trimeshes(scene)
#         trimeshes = [t.apply_transform(root_pose) for t in trimeshes]
#         RerunViewer.add_mesh_list_trimesh(self.name + name_suffix, trimeshes)
#         return

#     def _update_joints(self, joint_state: list | np.ndarray):
#         assert len(joint_state) == len(
#             self.urdf.actuated_joints
#         ), "Wrong number of joint values."
#         self.urdf.update_cfg(joint_state)
#         return

#     def _scene_to_trimeshes(self, scene: trimesh.Scene) -> list[trimesh.Trimesh]:
#         """
#         Convert a trimesh.Scene to a list of trimesh.Trimesh.

#         Skips objects that are not an instance of trimesh.Trimesh.
#         """
#         trimeshes = []
#         scene_dump = scene.dump()
#         geometries = [scene_dump] if not isinstance(scene_dump, list) else scene_dump
#         for geometry in geometries:
#             if isinstance(geometry, trimesh.Trimesh):
#                 trimeshes.append(geometry)
#             elif isinstance(geometry, trimesh.Scene):
#                 trimeshes.extend(self._scene_to_trimeshes(geometry))
#         return trimeshes
