import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class oxford_reader:
    def __init__(self, dataset_path):
        # /media/lair/ssd1/GS/Oxford Spires/bodleian-library-02
        self.dataset_path = dataset_path
        
        # User-provided calibration.
        # T_base_lidar_t_xyz_q_xyzw = [x, y, z, qx, qy, qz, qw]
        self.calib_val = [0.0, 0.0, 0.124, 0.0, 0.0, 1.0, 0.0]
        self.read_calibration_info()
        
        self.pointcloud_path = os.path.join(self.dataset_path, "lidar-clouds")
        pc_file_list = [f for f in os.listdir(self.pointcloud_path) if f.endswith('.pcd')]
        self.pc_file_list = sorted(pc_file_list)
        self.dataset_length = len(self.pc_file_list)
        
        self.read_timestamps()
        
        self.pose_file = os.path.join(self.dataset_path, "gt-tum.txt")
        self.gt_poses = self.load_pose()
        self.gt_poses_vis = np.array([x[:3, 3] for x in self.gt_poses])
        
        self.min_depth = 0.5 
        self.max_depth = 60.0 
        self.downsample_rate = 10
        self.data_counter = 0

    def read_calibration_info(self):
        x, y, z, qx, qy, qz, qw = self.calib_val
        self.velo_to_world = np.eye(4)
        self.velo_to_world[:3, 3] = [x, y, z]
        self.velo_to_world[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        self.velo_to_world_R = self.velo_to_world[:3, :3]
        self.velo_to_world_t = self.velo_to_world[:3, 3].T
        self.world_to_velo = np.linalg.inv(self.velo_to_world)

    def read_timestamps(self):
        self.timestamps = np.array([float(f.replace('.pcd', '')) for f in self.pc_file_list])

    def load_pose(self):
        gt_poses = []
        if not os.path.exists(self.pose_file):
            print(f"[Warning] GT file not found: {self.pose_file}")
            return [np.eye(4)] * self.dataset_length

        with open(self.pose_file, "r") as f:
            lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
            
            gt_data = np.array([list(map(float, l.split())) for l in lines])
            gt_ts = gt_data[:, 0]
            
            for ts in self.timestamps:
                idx = np.abs(gt_ts - ts).argmin()
                data = gt_data[idx]
                
                # Base_to_World Pose
                base_pose = np.eye(4)
                base_pose[:3, 3] = data[1:4]
                base_pose[:3, :3] = R.from_quat(data[4:8]).as_matrix()
                
                # LiDAR_to_World = Base_to_World * LiDAR_to_Base
                gt_poses.append(base_pose @ self.velo_to_world)
                
        return np.array(gt_poses)

    def load_points(self, iter):
        path = os.path.join(self.pointcloud_path, self.pc_file_list[iter])
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        
        intensity = np.ones(len(points)) * 0.5
        
        points_norm = np.linalg.norm(points, axis=-1)
        point_mask = (points_norm < self.max_depth) & (points_norm > self.min_depth)
        
        self.np_pcd = points[point_mask]
        self.np_intensity = intensity[point_mask]
        self.z_vals = points_norm[point_mask]

    def sequential_load(self, iter):
        self.load_points(iter)
        return self.np_pcd, self.np_intensity, self.z_vals

    def test_points(self):
        self.load_points(0)
        return self.np_pcd, self.np_intensity
    
    def save_traj(self, iter, poses):
        traj = np.array([x[:3, 3] for x in poses])
        plt.clf()
        plt.plot(traj[:, 0], traj[:, 1], label='Estimated')
        plt.plot(self.gt_poses_vis[:, 0], self.gt_poses_vis[:, 1], label='GT', alpha=0.5)
        plt.legend()
        plt.axis('equal')
        plt.savefig(os.path.join(self.dataset_path, "traj_result.png"))
