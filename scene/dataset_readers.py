from abc import ABC, abstractmethod
import os
import numpy as np
from typing import Tuple
from pathlib import Path
from plyfile import PlyData, PlyElement

from utils.config_utils import DatasetConfig, DatasetType
from utils.pointcloud_utils import (
    pointcloud_reader_available,
    PointCloudReader,
    PointCloudReader_BIN,
    PointCloudReader_ROSBAG,
    PointCloudReader_PCD
)
from utils.trajectory_utils import (
    trajectory_reader_available,
    TrajectoryReader,
    TrajectoryReader_KITTI,
    TrajectoryReader_TUM,
    TrajectoryReader_VILENS,
    TrajectoryReader_NULL
)

class DatasetReader(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset_path = config.dataset_path
        self.cloud_reader: PointCloudReader = None
        self.traj_reader: TrajectoryReader = None
        self.timestamps = []
        
    def __len__(self):
        if self.cloud_reader:
            return len(self.cloud_reader)
        return 0
        
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: points (N,3), intensity (N,), z_values (N,)
        """
        # Hack to access random index if reader supports it or if we just iterate
        # Since Splat-LOAM readers are iterative, we might need to jump if idx != current
        # But for sequential SLAM, idx usually increases.
        # If random access is needed, we need better reader support.
        # For now, let's assume we can access filenames list in cloud_reader.
        
        if hasattr(self.cloud_reader, 'filenames'):
            filename = self.cloud_reader.filenames[idx]
            # Some readers might need internal state update
            self.cloud_reader.current_index = idx + 1 # Advance?
            
            # Read Raw
            raw_data = self.cloud_reader.read_cloud(filename)
            # raw_data is (N, 3) or (N, 4)
            points_ts = None
            
            return self._process_cloud(raw_data, points_ts)
        
        elif hasattr(self.cloud_reader, 'bag'):
            # ROSBAG random access is slow/hard.
            # Warning: Random access on ROSBAG is linear scan usually unless indexed.
            # We assume sequential access for SLAM. 
            # If idx != current, we might fail or need to seek.
            # But here we just call next().
            
            # For now, rely on __next__ logic wrapper
            # We bypass index and just get next()
            xyz_i, points_ts, timestamp = next(self.cloud_reader)

            
            return self._process_cloud(xyz_i, points_ts)
            
           

        return np.zeros((0,3)), np.zeros(0), np.zeros(0), np.zeros(0) # Empty return for unsupported readers

    
    
    def _process_cloud(self, raw_data, points_ts):
        if raw_data.shape[1] >= 4:
            points = raw_data[:, :3]
            intensity = raw_data[:, 3]
        else:
            points = raw_data[:, :3]
            intensity = np.ones(points.shape[0]) * 0.5 # Default intensity
        
        if self.config.dataset_type == DatasetType.kitti:
            points = self.intrinsic_correct(points, correct_deg=0.195)  
            
        points_norm = np.linalg.norm(points, axis=1)
        
        # Filter by depth
        min_d = self.config.min_depth
        max_d = self.config.max_depth
        mask = (points_norm > min_d) & (points_norm < max_d)
        if points_ts is None:
            return points[mask], intensity[mask], points_norm[mask], None
        
        return points[mask], intensity[mask], points_norm[mask], points_ts[mask]
    
    def intrinsic_correct(self,points: np.ndarray, correct_deg=0.195):

        # # This function only applies for the KITTI dataset, and should NOT be used by any other dataset,
        # # the original idea and part of the implementation is taking from CT-ICP(Although IMLS-SLAM
        # # Originally introduced the calibration factor)
        # We set the correct_deg = 0.195 deg for KITTI odom dataset, inline with MULLS #issue 11
        if correct_deg == 0.0:
            return points

        dist = np.linalg.norm(points[:,:3],axis=1)
        dist_safe = np.clip(dist, 1e-12, None)

        kitti_var_vertical_ang = np.deg2rad(correct_deg)
        v_ang = np.arcsin(np.clip(points[:, 2] / dist_safe, -1.0, 1.0))
        v_ang_c = v_ang + kitti_var_vertical_ang

        cos_v = np.cos(v_ang)
        hor_scale = np.ones_like(cos_v)
        valid = np.abs(cos_v) > 1e-8
        hor_scale[valid] = np.cos(v_ang_c[valid]) / cos_v[valid]

        points[:, 0] *= hor_scale
        points[:, 1] *= hor_scale
        points[:, 2] = dist_safe * np.sin(v_ang_c)

        return points


    def get_pose(self, idx):
        # We need timestamp for pose usually
        # But if index-based (KITTI), we use index
        
        # 1. Try index based access if available (KITTI)
        if isinstance(self.traj_reader, TrajectoryReader_KITTI):
             if hasattr(self.traj_reader, 'poses') and len(self.traj_reader.poses) > idx:
                 return self.traj_reader.poses[idx] @ self.traj_reader.gt_T_s
             else:
                 return None

        # 2. Check if it is NULL reader
        if isinstance(self.traj_reader, TrajectoryReader_NULL):
            return None
            
        # 3. If Time based, we need timestamp
        # Retrieve timestamp from cloud reader
        if hasattr(self.cloud_reader, 'get_timestamp') and hasattr(self.cloud_reader, 'filenames'):
            # Check bounds for filenames just in case
            if idx < len(self.cloud_reader.filenames):
                ts = self.cloud_reader.get_timestamp(self.cloud_reader.filenames[idx])
                try:
                    return self.traj_reader(ts)
                except Exception as e:
                    print(f"[DEBUG] get_pose failed (filenames): {e}")
                    return None
            else:
                return None

        # If only a bag reader is available, fetch its timestamp before querying poses.
        elif hasattr(self.cloud_reader, 'get_timestamp'):
            ts = self.cloud_reader.get_timestamp(idx)
            try:
                return self.traj_reader(ts)
            except Exception as e:
                print(f"[DEBUG] get_pose failed (timestamp {ts}): {e}")
                pass # This will fall through to return None
        
        print("please check : timestamp_dtol")
        return None

        

    def get_timestamp(self, idx):

        if hasattr(self.cloud_reader, 'get_timestamp') and hasattr(self.cloud_reader, 'filenames'):

             return self.cloud_reader.get_timestamp(self.cloud_reader.filenames[idx])

        # ROSBAG case?

        return 0.0



    @property

    def gt_poses(self):

        if self.traj_reader and hasattr(self.traj_reader, 'poses') and len(self.traj_reader.poses) > 0:

            # Apply calibration gt_T_s to all poses

            return [pose @ self.traj_reader.gt_T_s for pose in self.traj_reader.poses]

        return []



class DatasetReader_KITTI(DatasetReader):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        pc_cfg = config.cloud_reader
        tr_cfg = config.trajectory_reader
        
        base_folder = Path(config.dataset_path)
        
        # Auto-configure if paths are defaults
        if pc_cfg.cloud_folder is None or pc_cfg.cloud_folder == "":
            if "velodyne" in base_folder.name:
                pc_cfg.cloud_folder = str(base_folder)
            else:
                pc_cfg.cloud_folder = str(base_folder / "velodyne")

        if pc_cfg.timestamp_filename is None:
            if "velodyne" in base_folder.name:
                 pc_cfg.timestamp_filename = str(base_folder.parent / "times.txt")
            else:
                 pc_cfg.timestamp_filename = str(base_folder / "times.txt")
        
        if tr_cfg.gt_T_sensor_kitti_filename is None:
            if "velodyne" in base_folder.name:
                tr_cfg.gt_T_sensor_kitti_filename = str(base_folder.parent / "calib.txt")
            else:
                tr_cfg.gt_T_sensor_kitti_filename = str(base_folder / "calib.txt")
            
        self.cloud_reader = PointCloudReader_BIN(pc_cfg)
        
        if tr_cfg.filename is None or not os.path.exists(tr_cfg.filename):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            tr_cfg.timestamp_from_filename_kitti = pc_cfg.timestamp_filename
            self.traj_reader = TrajectoryReader_KITTI(tr_cfg)

class DatasetReader_VBR(DatasetReader):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        pc_cfg = config.cloud_reader
        tr_cfg = config.trajectory_reader
        
        # VBR uses rosbag
        if pc_cfg.cloud_folder is None or pc_cfg.cloud_folder == "":
            pc_cfg.cloud_folder = config.dataset_path
            
        if pc_cfg.rosbag_topic is None:
            pc_cfg.rosbag_topic = "/ouster/points"
            
        self.cloud_reader = PointCloudReader_ROSBAG(pc_cfg)
        
        if tr_cfg.gt_T_sensor_t_xyz_q_xyzw is None:
            tr_cfg.gt_T_sensor_t_xyz_q_xyzw = [0, 0, 0, 0, 0, 0, 1]
            
        if tr_cfg.filename is None or not os.path.exists(tr_cfg.filename):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            self.traj_reader = TrajectoryReader_TUM(tr_cfg)

class DatasetReader_NCD(DatasetReader):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        pc_cfg = config.cloud_reader
        tr_cfg = config.trajectory_reader
        
        if pc_cfg.cloud_folder is None or pc_cfg.cloud_folder == "":
            pc_cfg.cloud_folder = config.dataset_path
            
        if pc_cfg.rosbag_topic is None:
            pc_cfg.rosbag_topic = "/os_cloud_node/points"
            
        self.cloud_reader = PointCloudReader_ROSBAG(pc_cfg)
        # NCD Calibration
        if tr_cfg.gt_T_sensor_t_xyz_q_xyzw is None:
            tr_cfg.gt_T_sensor_t_xyz_q_xyzw = [0.001, 0, 0.091, 0, 0, 0, 1]
        
        if tr_cfg.filename is None or not os.path.exists(tr_cfg.filename):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            self.traj_reader = TrajectoryReader_TUM(tr_cfg)

class DatasetReader_OXSPIRES(DatasetReader):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        pc_cfg = config.cloud_reader
        tr_cfg = config.trajectory_reader
        
        if pc_cfg.cloud_folder is None or pc_cfg.cloud_folder == "":
            pc_cfg.cloud_folder = config.dataset_path
            
        if pc_cfg.rosbag_topic is None:
             pc_cfg.rosbag_topic = "/hesai/pandar"
             
        self.cloud_reader = PointCloudReader_ROSBAG(pc_cfg)
        # Oxford Spires Calibration
        if tr_cfg.gt_T_sensor_t_xyz_q_xyzw is None:
            tr_cfg.gt_T_sensor_t_xyz_q_xyzw = [0, 0, 0.124, 0, 0, 1, 0]
        
        if tr_cfg.filename is None or not os.path.exists(tr_cfg.filename):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            self.traj_reader = TrajectoryReader_TUM(tr_cfg)

class DatasetReader_GENERIC(DatasetReader):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        pc_cfg = config.cloud_reader
        tr_cfg = config.trajectory_reader
        
        # Generic mix and match
        # Must ensure cloud_format is set in config
        if pc_cfg.cloud_format in pointcloud_reader_available:
             self.cloud_reader = pointcloud_reader_available[pc_cfg.cloud_format](pc_cfg)
        else:
             raise ValueError(f"Unknown cloud format: {pc_cfg.cloud_format}")

        if tr_cfg.reader_type in trajectory_reader_available:
             self.traj_reader = trajectory_reader_available[tr_cfg.reader_type](tr_cfg)
        else:
             self.traj_reader = TrajectoryReader_NULL(tr_cfg)


dataset_reader_factory = {
    DatasetType.kitti: DatasetReader_KITTI,
    DatasetType.vbr: DatasetReader_VBR,
    DatasetType.ncd: DatasetReader_NCD,
    DatasetType.oxspires: DatasetReader_OXSPIRES,
    DatasetType.generic: DatasetReader_GENERIC
}

def get_dataset_reader(config: DatasetConfig) -> DatasetReader:
    if config.dataset_type in dataset_reader_factory:
        return dataset_reader_factory[config.dataset_type](config)
    else:
        raise NotImplementedError(f"Dataset type {config.dataset_type} is not supported.")

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
