import numpy as np
import struct
import os
import matplotlib.pyplot as plt

class velodyne_reader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
        # calibration info (https://medium.com/@jaimin-k/exploring-kitti-visual-ododmetry-dataset-8ac588246cdc)
        self.read_calibration_info()
        
        # times (for visualization)
        self.read_timestamps()
        
        # pose txt file
        self.pose_file = os.path.join(dataset_path, "pose.txt")
        self.gt_poses = self.load_pose()
        self.gt_poses_vis = np.array([x[:3, 3] for x in self.gt_poses])
        
        # LiDAR point cloud files
        self.pointcloud_path = os.path.join(dataset_path, "velodyne")
        pc_file_list = os.listdir(self.pointcloud_path)
        self.pc_file_list = sorted(pc_file_list)
        self.dataset_length = len(self.pc_file_list)
        self.min_depth = 5
        self.max_depth = 40
        
        self.downsample_rate = 10
        
        self.data_counter = 0
    
    def read_timestamps(self):
        with open(os.path.join(self.dataset_path, "times.txt"), "rb") as f:
            lines = f.readlines()
            self.timestamps = []
            for time in lines:
                self.timestamps.append(float(time))
        self.timestamps = np.asarray(self.timestamps)
        
    def read_calibration_info(self):
        with open(os.path.join(self.dataset_path, "calib.txt"), "rb") as f:
            lines = f.readlines()
            velo_to_world_info = lines[4].split()
            self.velo_to_world = np.asarray(velo_to_world_info[1:13]).astype(np.float32)
            self.velo_to_world = np.concatenate((self.velo_to_world,
                                            [0,0,0,1])).reshape(4,4)
            self.velo_to_world_R = self.velo_to_world[:3,:3]
            self.velo_to_world_t = self.velo_to_world[:3,3].T
            self.world_to_velo = np.linalg.inv(self.velo_to_world)
    
    def test_points(self):
        with open (os.path.join(self.pointcloud_path, self.pc_file_list[0]), "rb") as f:
            size_float = 4
            byte = f.read(size_float*4)
            list_pcd = []
            list_intensity = []
            counter = 0
            while byte:
                if counter % self.downsample_rate == 0:
                    x,y,z,intensity = struct.unpack("ffff", byte)
                    list_pcd.append([x, y, z])
                    list_intensity.append(intensity)
                    byte = f.read(size_float*4)
                counter += 1
        points = np.asarray(list_pcd)
        
        self.np_pcd = points
        self.np_intensity = np.asarray(list_intensity)
        return self.np_pcd, self.np_intensity
    
    def load_pose(self):
        gt_poses = []
        with open(self.pose_file, "rb") as f:
            lines = f.readlines()
        for line in lines:
            pose = np.eye(4)
            pose[:3,:4] = np.array(list(map(float, line.split()))).reshape(3,4)
            # pose = np.linalg.inv(pose)
            pose = pose@self.velo_to_world
            gt_poses.append(pose)
        gt_poses_np = np.array(gt_poses)
        return gt_poses_np
        
    def sequential_load(self, iter):
        self.load_points(iter)
        return self.np_pcd, self.np_intensity, self.z_vals
        
    def load_points(self, iter):
        path = os.path.join(self.pointcloud_path, self.pc_file_list[iter])
        with open (path, "rb") as f:
            size_float = 4
            byte = f.read(size_float*4)
            list_pcd = []
            list_intensity = []
            counter = 0
            while byte:
                # if counter % self.downsample_rate == 0:
                x,y,z,intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                list_intensity.append(intensity)
                byte = f.read(size_float*4)
                counter += 1
            
        points = np.asarray(list_pcd)
        points_norm = np.linalg.norm(points, axis=-1)
        # print(points.shape)
        point_mask = True
        point_mask = (points_norm < self.max_depth) & point_mask
        point_mask = (points_norm > self.min_depth) & point_mask
        
        self.np_pcd = points[point_mask]
        self.np_intensity = np.asarray(list_intensity)[point_mask]
        self.z_vals = points_norm[point_mask]
        
    def plot_traj(self, iter, poses):
        '''
        Plot trajectory
        
        iter : iter
        poses : list of estimated poses
        '''
        iter += 1
        traj = np.array([x[:3, 3] for x in poses])
        plt.clf()
        plt.title(f'{iter}')
        plt.plot(traj[:, 0], traj[:, 2], label='estimated trajectory', linewidth=3)
        plt.legend()
        plt.plot(self.gt_poses_vis[:iter, 0], self.gt_poses_vis[:iter, 2], label='g.t. trajectory')
        plt.legend()
        plt.axis('equal')
        plt.pause(1e-15)
        
    def save_traj(self, iter, poses):
        '''
        save trajectory
        
        iter : iter
        poses : list of estimated poses
        '''
        iter += 1
        traj = np.array([x[:3, 3] for x in poses])
        plt.clf()
        plt.title(f'{iter}')
        plt.plot(traj[:, 0], traj[:, 2], label='estimated trajectory', linewidth=3)
        plt.legend()
        plt.plot(self.gt_poses_vis[:, 0], self.gt_poses_vis[:, 2], label='g.t. trajectory')
        plt.legend()
        plt.axis('equal')
        plt.savefig("traj_result.png")
