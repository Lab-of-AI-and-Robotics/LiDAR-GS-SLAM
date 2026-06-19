import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from rosbags.typesys import stores, get_types_from_msg, get_typestore

class newer_reader:
    def __init__(self, dataset_path, bag_name=""):
        # Example: /media/lair/ssd1/NewerCollege/2021-01-26-11-20-41
        self.dataset_path = dataset_path
        
        # T_lidar_t_xyz_q_xyzw = [x, y, z, qx, qy, qz, qw]
        self.calib_val = [0.001, 0.0, 0.091, 0.0, 0.0, 0.0, 1.0]
        self.read_calibration_info()
        
        if bag_name:
            self.bag_path = Path(os.path.join(self.dataset_path, bag_name))
        else:
            # Automatically choose the first bag file in the dataset directory.
            bags = list(Path(self.dataset_path).glob("*.bag"))
            if not bags:
                raise FileNotFoundError(f"No .bag files found in {dataset_path}")
            self.bag_path = bags[0]

        print(f"[Loader] Opening ROSBag: {self.bag_path}")
        self.reader = AnyReader([self.bag_path])
        self.reader.open()
        
        self.topic = "/os_cloud_node/points" 
        self.connections = [x for x in self.reader.connections if x.topic == self.topic]
        if not self.connections:
            avail = [x.topic for x in self.reader.connections]
            raise ValueError(f"Topic {self.topic} not found. Available: {avail}")
        
        self.dataset_length = self.reader.topics[self.topic].msgcount
        
        print("[Loader] Scanning bag for timestamps... (this might take a moment)")
        self.timestamps = []
        self.msg_gen = self.reader.messages(connections=self.connections)
        for conn, timestamp, rawdata in self.reader.messages(connections=self.connections):
            # ROS time (nanoseconds) -> seconds
            self.timestamps.append(timestamp / 1e9)
        self.timestamps = np.array(self.timestamps)
        
        # Load GT poses in TUM format (gt-tum.txt).
        self.pose_file = os.path.join(self.dataset_path, "gt-tum.txt") 
        self.gt_poses = self.load_pose()
        self.gt_poses_vis = np.array([x[:3, 3] for x in self.gt_poses])

        self.min_depth = 0.5 
        self.max_depth = 120.0
        self.reset_generator()
        self.current_iter = -1

    def reset_generator(self):
        self.msg_gen = self.reader.messages(connections=self.connections)
        self.current_iter = -1

    def read_calibration_info(self):
        x, y, z, qx, qy, qz, qw = self.calib_val
        self.gt_to_lidar = np.eye(4)
        self.gt_to_lidar[:3, 3] = [x, y, z]
        self.gt_to_lidar[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        self.lidar_to_gt = np.linalg.inv(self.gt_to_lidar)

    def load_pose(self):
        gt_poses = []
        if not os.path.exists(self.pose_file):
            print(f"[Warning] GT file not found: {self.pose_file}")
            return [np.eye(4)] * self.dataset_length

        print(f"[Loader] Loading GT from {self.pose_file}")
        with open(self.pose_file, "r") as f:
            lines = [line.strip() for line in f if not line.startswith("#") and line.strip()]
            
            gt_data = np.array([list(map(float, l.split())) for l in lines])
            gt_ts = gt_data[:, 0]
            
            # Timestamp Matching
            for ts in self.timestamps:
                idx = np.abs(gt_ts - ts).argmin()
                data = gt_data[idx]
                
                # World_to_Body (GT Base) Pose
                base_pose = np.eye(4)
                base_pose[:3, 3] = data[1:4]
                base_pose[:3, :3] = R.from_quat(data[4:8]).as_matrix() # xyzw
                
                # World_to_LiDAR = World_to_Body * Body_to_LiDAR
                gt_poses.append(base_pose @ self.gt_to_lidar)
                
        return np.array(gt_poses)

    def parse_point_cloud(self, raw_data, msg_type):
        msg = self.reader.deserialize(raw_data, msg_type)
        
        width = msg.width
        height = msg.height
        point_step = msg.point_step
        row_step = msg.row_step
        data = msg.data
                
        # Convert raw data to a flat uint8 array.
        raw_arr = np.frombuffer(data, dtype=np.uint8)
        
        dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        
        has_intensity = False
        intensity_offset = -1
        for field in msg.fields:
            if field.name == 'intensity':
                has_intensity = True
                intensity_offset = field.offset
                break
        float_data = raw_arr.view(np.float32)
        stride = int(point_step / 4)
        
        x = float_data[0::stride]
        y = float_data[1::stride]
        z = float_data[2::stride]
        
        points = np.stack([x, y, z], axis=-1)
        
        if has_intensity:
            i_stride_offset = int(intensity_offset / 4)
            intensity = float_data[i_stride_offset::stride]
        else:
            intensity = np.zeros(len(points))

        return points, intensity

    def load_points(self, iter):
        if iter != self.current_iter + 1:
            if iter == 0:
                self.reset_generator()
            elif iter > self.current_iter + 1:
                while self.current_iter < iter - 1:
                    next(self.msg_gen)
                    self.current_iter += 1
            else:
                self.reset_generator()
                while self.current_iter < iter - 1:
                    next(self.msg_gen)
                    self.current_iter += 1
        try:
            conn, timestamp, raw_data = next(self.msg_gen)
            self.current_iter += 1
        except StopIteration:
            print("[Loader] End of bag reached.")
            return

        points, intensity = self.parse_point_cloud(raw_data, conn.msgtype)
        
        # Range filtering.
        points_norm = np.linalg.norm(points, axis=-1)
        point_mask = (points_norm < self.max_depth) & (points_norm > self.min_depth)
        
        self.np_pcd = points[point_mask]
        self.np_intensity = intensity[point_mask]
        self.z_vals = points_norm[point_mask]

    def sequential_load(self, iter):
        self.load_points(iter)
        return self.np_pcd, self.np_intensity, self.z_vals

    def test_points(self):
        # Reset to frame 0 for testing.
        self.load_points(0)
        return self.np_pcd, self.np_intensity
    
    def save_traj(self, iter, poses):
        """Save trajectory visualization."""
        traj = np.array([x[:3, 3] for x in poses])
        plt.clf()
        plt.plot(traj[:, 0], traj[:, 1], label='Estimated')
        plt.plot(self.gt_poses_vis[:, 0], self.gt_poses_vis[:, 1], label='GT', alpha=0.5)
        plt.legend()
        plt.axis('equal')
        plt.title(f"Trajectory ~ Frame {iter}")
        plt.savefig(os.path.join(self.dataset_path, "traj_result.png"))

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'reader' in state:
            del state['reader']
        if 'msg_gen' in state:
            del state['msg_gen']
        if 'connections' in state:
            del state['connections']
        return state

    # Reopen the bag and restore the generator after unpickling.
    def __setstate__(self, state):
        self.__dict__.update(state)
        
        # Reopen the bag in the new process.
        self.reader = AnyReader([self.bag_path])
        self.reader.open()
        
        # Restore connection metadata.
        self.connections = [x for x in self.reader.connections if x.topic == self.topic]
        
        # Recreate the generator and reset the index for the newly opened file.
        self.msg_gen = self.reader.messages(connections=self.connections)
        self.current_iter = -1

    def __del__(self):
        if hasattr(self, 'reader'):
            self.reader.close()
