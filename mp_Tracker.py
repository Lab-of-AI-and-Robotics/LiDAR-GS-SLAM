import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import cv2
import numpy as np
import pygicp
import time
from scipy.spatial.transform import Rotation
import rerun as rr
from matplotlib import cm
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.lab_utils import rgb2lab_np
from scene.dataset_readers import get_dataset_reader
from utils.graphics_utils import getWorld2View2
from gaussian_renderer import render, network_gui
from tqdm import tqdm
from datetime import datetime
import small_gicp
import pyprojections as pyp
import roma


class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.cfg = slam.cfg
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = slam.keyframe_th
        self.knn_max_distance = slam.knn_max_distance
        self.overlapped_th = slam.overlapped_th
        self.overlapped_th2 = slam.overlapped_th2
        self.downsample_rate = slam.downsample_rate
        self.downsample_voxel_size = slam.downsample_voxel_size
        self.loop_constraint_noise = slam.loop_constraint_noise
        self.test = slam.test
        self.use_dynamic_fov = slam.use_dynamic_fov
        self.last_kf_pos = np.zeros(3)
        
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.rerun_viewer = slam.rerun_viewer
        
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.reg = pygicp.FastGICP()
        
        self.dataloader = None
        self.poses = []
        self.gt_poses_matched = []
        self.tracking_ate_rmse_shared = slam.tracking_ate_rmse_shared
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        
        self.train_iter = 0
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
        self.tracking_fps = 0.0
        self.tracking_avg_fps_shared = slam.tracking_avg_fps_shared
        
        self.current_pose_shared = slam.current_pose_shared
        self.iter_shared = slam.iter_shared
        self.gradient_png_max_frames = 100
        self.gradient_png_dir = os.path.join(self.output_path, "gradient_debug")
        os.makedirs(self.gradient_png_dir, exist_ok=True)
        self.edge_focus_w_depth = float(getattr(slam, "edge_focus_w_depth", 1.0))
        self.edge_focus_w_normal = float(getattr(slam, "edge_focus_w_normal", 1.0))
        self.edge_focus_q_low = float(getattr(slam, "edge_focus_q_low", 5.0))
        self.edge_focus_q_high = float(getattr(slam, "edge_focus_q_high", 95.0))
        self.edge_focus_smooth_kernel = int(getattr(slam, "edge_focus_smooth_kernel", 3))
        self.edge_focus_invalid_erosion = int(getattr(slam, "edge_focus_invalid_erosion", 1))

    def make_range_image(self, cloud, image_height, image_width):
        pts = np.asarray(cloud, dtype=np.float32)
        K,_,vfov,hfov = pyp.calculate_spherical_intrinsics(pts.T,image_height, image_width)
        if(self.use_dynamic_fov == True):
            self.fx = K[0,0]
            self.fy = K[1,1]
            self.cx = K[0,2]
            self.cy = K[1,2]
        projector = pyp.Camera(image_height,image_width,K,0,self.depth_trunc,pyp.CameraModel.Spherical)
        lut, _ = projector.project(pts.T)
        
        depth = np.zeros((image_height, image_width), dtype=np.float32)
        valid_mask = (lut != -1)
        valid_pixel_indices = np.where(valid_mask)
        valid_point_indices = lut[valid_mask]

        ranges = np.linalg.norm(cloud[valid_point_indices], axis=1)
        depth[valid_pixel_indices] = ranges

        return depth, valid_mask, K
    
    def make_normal_image(self,
        cloud,
        gicp_rotations,
        gicp_scales,
        R_c2w,
        t_c2w,
        rotation_frame="sensor",
        planarity_threshold=0.1,
        grazing_threshold=0.05,
    ):
        image_height = self.H
        image_width = self.W
        pts = np.asarray(cloud, dtype=np.float32)
        K, _, vfov, hfov = pyp.calculate_spherical_intrinsics(pts.T, image_height, image_width)
        projector = pyp.Camera(image_height, image_width, K, 0, self.depth_trunc, pyp.CameraModel.Spherical)
        lut, _ = projector.project(pts.T)
        normal_image = np.zeros((image_height, image_width, 3), dtype=np.float32)
        rots_scipy = Rotation.from_quat(gicp_rotations)  # xyzw
        rot_matrices = rots_scipy.as_matrix()  # (N, 3, 3)
        scales = gicp_scales
        normals_from_rot = np.array([rot_matrices[i, :, np.argmin(scales[i])]
                                     for i in range(len(scales))])
        if rotation_frame == "sensor":
            normals_sensor = normals_from_rot
        elif rotation_frame == "world":
            normals_sensor = normals_from_rot @ R_c2w
        else:
            raise ValueError(f"Unknown rotation_frame: {rotation_frame}")

        ray_direction = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-6)

        dot_products = np.sum(normals_sensor * ray_direction, axis=1)

        normals_sensor[dot_products > 0] *= -1

        # Confidence mask
        sorted_scales = np.sort(gicp_scales, axis=1)
        planarity = 1.0 - (sorted_scales[:, 0] / (sorted_scales[:, 1] + 1e-6))

        # Optionally reject grazing-angle observations.
        grazing_mask = np.abs(dot_products) < grazing_threshold

        confidence_mask = (planarity > planarity_threshold) & (~grazing_mask)

        # Transform normals from sensor to world frame.
        normals_world = normals_sensor @ R_c2w.T
        normals_world[~confidence_mask] = 0

        # Rasterize
        valid_mask = (lut != -1)
        valid_pixel_indices = np.where(valid_mask)
        valid_point_indices = lut[valid_mask]
        normal_image[valid_pixel_indices] = normals_world[valid_point_indices]

        return normal_image.transpose(2, 0, 1)
        
    def run(self):
        self.dataloader = get_dataset_reader(self.cfg.data)
        
        init_pose = self.dataloader.get_pose(0)
        if init_pose is None:
            self.poses = [np.eye(4)]
            self.gt_poses_matched = []
        else:
            self.poses = [init_pose]
            self.gt_poses_matched = [init_pose]
            
        self.tracking()
    def _compute_gicp_features(self, scales, corr_idx, corr_sq, points):
        scales = np.asarray(scales, dtype=np.float32).reshape(-1, 3)
        corr_idx = np.asarray(corr_idx).reshape(-1)
        corr_sq = np.asarray(corr_sq, dtype=np.float32).reshape(-1)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)

        n = min(scales.shape[0], corr_idx.shape[0], corr_sq.shape[0], points.shape[0])
        if n <= 0:
            return None

        scales = scales[:n]
        corr_idx = corr_idx[:n]
        corr_sq = corr_sq[:n]
        points = points[:n]

        lam = np.sort(scales * scales, axis=1)
        l0 = lam[:, 0]
        l1 = lam[:, 1]
        l2 = lam[:, 2]
        eps = 1e-6

        curv = l0 / (l0 + l1 + l2 + eps)
        planar = (l1 - l0) / (l2 + eps)
        linear = (l2 - l1) / (l2 + eps)
        density_proxy = 1.0 / (np.sqrt(np.maximum(l0 * l1 * l2, 0.0)) + eps)

        valid = corr_idx >= 0
        res = np.sqrt(np.clip(corr_sq, 0.0, None))
        res_n = np.zeros_like(res, dtype=np.float32)
        if np.any(valid):
            res_p95 = np.percentile(res[valid], 95)
            if res_p95 > 1e-8:
                res_n[valid] = np.clip(res[valid] / (res_p95 + eps), 0.0, 1.0)

        def robust_norm_1d(x, m):
            out = np.zeros_like(x, dtype=np.float32)
            vals = x[m]
            if vals.size < 16:
                return out
            lo = np.percentile(vals, self.edge_focus_q_low)
            hi = np.percentile(vals, self.edge_focus_q_high)
            if hi <= lo:
                return out
            out = np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)
            out[~m] = 0.0
            return out.astype(np.float32)

        planar_n = robust_norm_1d(planar, valid)
        curv_n = robust_norm_1d(curv, valid)
        linear_n = robust_norm_1d(linear, valid)
        density_n = robust_norm_1d(density_proxy, valid)
        range_m = np.linalg.norm(points, axis=1).astype(np.float32)
        range_n = robust_norm_1d(range_m, valid)

        plane_score = np.clip(planar_n * (1.0 - curv_n) * (1.0 - res_n), 0.0, 1.0)
        split_score = np.clip(0.5 * linear_n + 0.3 * curv_n + 0.2 * res_n, 0.0, 1.0)
        edge_score = np.maximum.reduce([linear_n, curv_n, res_n]).astype(np.float32)
        control_score = np.clip(0.55 * linear_n + 0.30 * curv_n + 0.15 * res_n, 0.0, 1.0).astype(np.float32)

        valid_vals = control_score[valid]
        if valid_vals.size >= 64:
            vals_u8 = np.clip(valid_vals * 255.0, 0.0, 255.0).astype(np.uint8).reshape(-1, 1)
            t_mid_u8, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t_mid = float(t_mid_u8) / 255.0
        elif valid_vals.size > 0:
            t_mid = float(np.percentile(valid_vals, 60.0))
        else:
            t_mid = 0.5
        t_plane = float(np.clip(t_mid - 0.05, 0.0, 1.0))
        t_edge = float(np.clip(t_mid + 0.05, 0.0, 1.0))

        # -1: invalid, 0: plane, 1: middle, 2: edge
        control_class = np.full((n,), -1, dtype=np.int8)
        if np.any(valid):
            plane_mask = valid & (control_score <= t_plane)
            edge_mask = valid & (control_score >= t_edge)
            middle_mask = valid & (~plane_mask) & (~edge_mask)
            control_class[plane_mask] = 0
            control_class[middle_mask] = 1
            control_class[edge_mask] = 2

        return {
            "valid": valid,
            "planar": planar.astype(np.float32),
            "curv": curv.astype(np.float32),
            "linear": linear.astype(np.float32),
            "density_proxy": density_proxy.astype(np.float32),
            "res": res.astype(np.float32),
            "res_n": res_n.astype(np.float32),
            "planar_n": planar_n,
            "curv_n": curv_n,
            "linear_n": linear_n,
            "density_n": density_n,
            "range_m": range_m,
            "range_n": range_n,
            "plane_score": plane_score.astype(np.float32),
            "split_score": split_score.astype(np.float32),
            "edge_score": edge_score,
            "control_score": control_score,
            "control_class": control_class,
            "t_plane": t_plane,
            "t_edge": t_edge,
            "t_mid": t_mid,
            }
    def compute_phys_conf(self, pts_sensor, rots_q, scales, R_c2w=None, rotation_frame="sensor"):

            pts_sensor = np.asarray(pts_sensor, dtype=np.float32).reshape(-1, 3)
            rots_q = np.asarray(rots_q, dtype=np.float32).reshape(-1, 4)
            scales = np.asarray(scales, dtype=np.float32).reshape(-1, 3)
            n = min(pts_sensor.shape[0], rots_q.shape[0], scales.shape[0])
            if n <= 0:
                return np.zeros((0,), dtype=np.float32)
            pts_sensor = pts_sensor[:n]
            rots_q = rots_q[:n]
            scales = scales[:n]

            eps = 1e-6
            rot_mats = Rotation.from_quat(rots_q).as_matrix()  # [N,3,3], xyzw
            min_axis = np.argmin(scales, axis=1)
            normals = rot_mats[np.arange(n), :, min_axis]

            if rotation_frame == "world" and R_c2w is not None:
                normals = normals @ np.asarray(R_c2w, dtype=np.float32)

            rays = pts_sensor / (np.linalg.norm(pts_sensor, axis=1, keepdims=True) + eps)
            cos_inc = np.abs(np.sum(normals * rays, axis=1))
            range_m = np.linalg.norm(pts_sensor, axis=1)

            r0 = 50.0
            cos0 = 0.1
            q_range = np.exp(-((range_m / r0) ** 2))
            q_inc = np.clip((cos_inc - cos0) / (1.0 - cos0 + eps), 0.0, 1.0)
            
            return np.clip(q_range * q_inc, 0.0, 1.0).astype(np.float32)

    def tracking(self):
        print("--- TRACKER (Process-0): Started successfully.")
        try:
            cmap = cm.get_cmap('viridis', 1000)
            self._init_downsample_stats()
            tt = torch.zeros((1,1)).float().cuda()
            self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
            self.reg.set_max_knn_distance(self.knn_max_distance)
            if_mapping_keyframe = False

            while not self.is_mapping_process_started[0]:
                time.sleep(0.01)

            self.total_start_time = time.time()
            pbar = tqdm(total=len(self.dataloader))

            for ii in range(len(self.dataloader)):                
                self.iter_shared[0] = ii
                raw_points, intensity, z_values, points_ts = self.dataloader[self.iteration_images]
                

                frame_start = time.perf_counter()
                
                if self.iteration_images == 0:
                    current_c2w = self.poses[-1]
                    self.last_kf_pos = current_c2w[:3, 3].copy()
                    self.current_pose_shared[:,:] = torch.from_numpy(current_c2w).cuda()
                   
                    depth_image, valid_mask, K = self.make_range_image(raw_points,self.H,self.W)
                    num_before = raw_points.shape[0]
                    pc = small_gicp.PointCloud(raw_points)
                    pc_ds = small_gicp.voxelgrid_sampling(pc, self.downsample_voxel_size)
                    
                   
                    pts_ds = pc_ds.points()[:,:3]
                    intensity = np.zeros(pts_ds.shape[0],dtype=np.float32)
                    z_values = np.linalg.norm(pts_ds, axis = 1)
                    trackable_filter = np.where(z_values!=0)[0]
                    self._accumulate_downsample_stats(num_before, pts_ds.shape[0], trackable_filter.shape[0])
                    

                    current_w2c = np.linalg.inv(current_c2w)
                    T = current_c2w[:3,3]
                    R = current_c2w[:3,:3]
                    points = (R @ pts_ds.T).T + T

                    self.reg.set_input_target(points)
                    num_trackable_points = trackable_filter.shape[0]
                    input_filter = np.zeros(points.shape[0], dtype=np.int32)
                    input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                    self.reg.set_target_filter(num_trackable_points, input_filter)
                    self.reg.calculate_target_covariance_with_filter()

                    rots = self.reg.get_target_rotationsq()
                    scales = self.reg.get_target_scales()
                    rots = np.reshape(rots, (-1,4))
                    scales = np.reshape(scales, (-1,3))
                    colors = cmap(intensity)[:,:3]
                    init_control_score = np.zeros((points.shape[0],), dtype=np.float32)
                    init_phys_conf = self.compute_phys_conf(pts_ds, rots, scales, current_c2w[:3,:3], rotation_frame="world")
                    
                    self.reg.set_target_control_score(init_control_score)                   
                    self.shared_new_gaussians.input_values(torch.tensor(points),  
                                                        torch.tensor(rots), torch.tensor(scales), 
                                                        torch.tensor(z_values), torch.tensor(trackable_filter),
                                                        torch.tensor(init_control_score),torch.tensor(init_phys_conf))
                    
                    depth_image, valid_mask, K = self.make_range_image(raw_points,self.H,self.W)
                    
                    
                    normal_image = self.make_normal_image(
                        pts_ds,
                        rots,
                        scales,
                        current_c2w[:3,:3],
                        current_c2w[:3,3],
                        rotation_frame="world"
                        )
                    self.shared_cam.setup_cam(current_w2c[:3,:3], current_w2c[:3,3], depth_image,K)
                    self.shared_cam.cam_idx[0] = self.iteration_images
                    
                    self.is_tracking_keyframe_shared[0] = 1
                    self.is_mapping_keyframe_shared[0]= 1
                    
                    while self.demo[0]:
                        time.sleep(0.001)
                        self.total_start_time = time.time() 
                
                else:
                    # Tracking
                    num_before = raw_points.shape[0]
                    if len(self.poses) >= 2:
                        T_last_cur = np.linalg.inv(self.poses[-2]) @ self.poses[-1]
                        raw_points = self.deskewing(raw_points, points_ts, T_last_cur)

                    depth_image, valid_mask, K = self.make_range_image(raw_points,self.H,self.W)
                    pc = small_gicp.PointCloud(raw_points)

                    pc_ds = small_gicp.voxelgrid_sampling(pc, self.downsample_voxel_size)
                    pts_ds = pc_ds.points()[:,:3]
                    intensity = np.zeros(pts_ds.shape[0],dtype=np.float32)
                    z_values = np.linalg.norm(pts_ds, axis = 1)
                    trackable_filter = np.where(z_values != 0)[0]
                    self._accumulate_downsample_stats(num_before, pts_ds.shape[0], trackable_filter.shape[0])
                    colors = cmap(intensity)[:, :3]

                    if len(self.poses) >= 2:
                        initial_pose = self.poses[-1]@np.linalg.inv(self.poses[-2])@self.poses[-1]
                    else:
                        initial_pose = self.poses[-1]                    
                    
                    
                    
                    self.reg.set_input_source(pts_ds)
                    num_trackable_points = trackable_filter.shape[0]
                    input_filter = np.zeros(pts_ds.shape[0], dtype=np.int32)
                    input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                    self.reg.set_source_filter(num_trackable_points, input_filter)
                    self.reg.set_source_intensities(intensity)
                
                    current_c2w = self.reg.align(initial_pose)
                    gt_pose = self.dataloader.get_pose(self.iteration_images)
                   
                    if gt_pose is not None:
                        self.gt_poses_matched.append(gt_pose)

                    self.poses.append(current_c2w)
                    
                    
                    
                    self.current_pose_shared[:,:] = torch.from_numpy(current_c2w).cuda()
                    current_w2c = np.linalg.inv(current_c2w)
                    T = current_c2w[:3,3]
                    R = current_c2w[:3,:3]
                    points = (R @ pts_ds.T).T + T

                    corr_idx, distances = self.reg.get_source_correspondence()
                    scales_for_control = np.array(self.reg.get_source_scales()).reshape((-1,3)) 
                    gicp_features_start_time = time.perf_counter()
                    feature_dict = self._compute_gicp_features(
                        scales_for_control,
                        np.asarray(corr_idx),
                        np.asarray(distances),
                        pts_ds,
                    )
                    if feature_dict is None:
                        control_score_frame = np.zeros((pts_ds.shape[0],), dtype=np.float32)
                    else:
                        control_score_frame = np.asarray(feature_dict["control_score"], dtype=np.float32)
                     
                    control_score_duration = time.perf_counter() - gicp_features_start_time

                    len_corres = len(np.where(distances<self.overlapped_th)[0])
                    
                    if (len_corres/distances.shape[0] < self.keyframe_th):
                        if_tracking_keyframe = True
                        self.from_last_tracking_keyframe = 0
                    else:
                        if_tracking_keyframe = False
                        self.from_last_tracking_keyframe += 1
                  
                    if_mapping_keyframe = (self.from_last_tracking_keyframe > 0) and (self.from_last_tracking_keyframe % self.keyframe_freq == 0)

                    if if_tracking_keyframe:
                      
                        while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                            time.sleep(1e-5)
                 
                        rots = np.array(self.reg.get_source_rotationsq())
                        rots = np.reshape(rots, (-1,4))
                        rots_local = np.array(self.reg.get_source_rotationsq())  # Keep rotations in the sensor frame.
                        rots_local = np.reshape(rots_local, (-1,4))
                     
                        R_d = Rotation.from_matrix(R)    # from camera R
                        R_d_q = R_d.as_quat() 
                        
                        rots = self.quaternion_multiply(R_d_q, rots)
                        scales = np.array(self.reg.get_source_scales())
                        scales = np.reshape(scales, (-1,3))
                       
                        phys_conf_start_time = time.perf_counter()
                        phys_conf_frame = self.compute_phys_conf(pts_ds, rots_local, scales, current_c2w[:3,:3], rotation_frame="world")
                        
                        phys_conf_duration = time.perf_counter() - phys_conf_start_time

                        intensity = np.clip(self.reg.get_source_intensities(), 0., 1.)
                        colors = cmap(intensity)[:,:3]
                        
                        not_overlapped = self.eliminate_overlapped2(distances, self.overlapped_th2)
                        trackable_filter = trackable_filter[not_overlapped]

                        capacity = self.shared_new_gaussians.xyz.shape[0]
                        used = int(self.shared_new_gaussians.using_idx[0])
                        remaining = capacity - used

                        if remaining <= 0:
                            return

                        N = min(points.shape[0], remaining)

                        points = points[:N]
                        colors = colors[:N]
                        rots = rots[:N]
                        scales = scales[:N]
                        z_values = z_values[:N]
                        trackable_filter = trackable_filter[:N]
                        control_score_frame = control_score_frame[:N]
                        phys_conf_frame = phys_conf_frame[:N]
                        self.shared_new_gaussians.input_values(torch.tensor(points), 
                                                        torch.tensor(rots), torch.tensor(scales), 
                                                        torch.tensor(z_values), torch.tensor(trackable_filter),
                                                        torch.tensor(control_score_frame), torch.tensor(phys_conf_frame))

                        depth_image, valid_mask, K = self.make_range_image(raw_points, self.H, self.W)
                        normal_image = self.make_normal_image(pts_ds, rots_local, scales, current_c2w[:3,:3], current_c2w[:3,3])
                        self.shared_cam.setup_cam(current_w2c[:3,:3], current_w2c[:3,3], depth_image,K)
                        self.shared_cam.cam_idx[0] = self.iteration_images
                        self.is_tracking_keyframe_shared[0] = 1
                        
                        while not self.target_gaussians_ready[0]:
                            time.sleep(1e-15)
                            
                        target_points, target_rots, target_scales, target_control_score = self.shared_target_gaussians.get_values_np()
                        self.reg.set_input_target(target_points)
                        self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
                        s = np.sort(target_scales, axis=1)
                        ratio = s[:, 2] / np.maximum(s[:, 1], 1e-6)
                        ratio = np.clip((ratio - 1.0)/3.0,0.0,1.0)
                        
                        tcs = ratio.flatten()
                      
                        self.reg.set_target_control_score(target_control_score.flatten())   
                        self.target_gaussians_ready[0] = 0
                        
                        
                        ts = self.dataloader.get_timestamp(self.iteration_images)
                    
                    elif if_mapping_keyframe:
                        while self.is_tracking_keyframe_shared[0] or self.is_mapping_keyframe_shared[0]:
                            time.sleep(1e-15)
                        
                        rots = np.array(self.reg.get_source_rotationsq())
                        rots = np.reshape(rots, (-1,4))
                        rots_local = np.array(self.reg.get_source_rotationsq())  # Keep rotations in the sensor frame.
                        rots_local = np.reshape(rots_local, (-1,4))

                        R_d = Rotation.from_matrix(R)    # from camera R
                        R_d_q = R_d.as_quat() 
                        rots = self.quaternion_multiply(R_d_q, rots)
                        scales = np.array(self.reg.get_source_scales())
                        scales = np.reshape(scales, (-1,3))
                        phys_conf_frame = self.compute_phys_conf(pts_ds, rots_local, scales, current_c2w[:3,:3], rotation_frame="world")
                        intensity = self.reg.get_source_intensities()
                        intensity = np.clip(intensity, 0., 1.)
                        intensity = intensity / 2.
                        colors = cmap(intensity)[:,:3]

                        self.shared_new_gaussians.input_values(torch.tensor(points), 
                                                        torch.tensor(rots), torch.tensor(scales), 
                                                        torch.tensor(z_values), torch.tensor(trackable_filter),
                                                        torch.tensor(control_score_frame),torch.tensor(phys_conf_frame))
                        
                        depth_image, valid_mask, K = self.make_range_image(raw_points, self.H, self.W)
                        normal_image = self.make_normal_image(pts_ds, rots_local, scales, current_c2w[:3,:3], current_c2w[:3,3])
                        self.shared_cam.setup_cam(current_w2c[:3,:3], current_w2c[:3,3], depth_image,K)
                        self.shared_cam.cam_idx[0] = self.iteration_images
                        self.is_mapping_keyframe_shared[0] = 1
                        
                        ts = self.dataloader.get_timestamp(self.iteration_images)
                
                pbar.update(1)
                FPS = 1.0 / (time.perf_counter() - frame_start)
                self.tracking_fps += FPS
                
                self.iteration_images += 1
                torch.cuda.empty_cache()
            
            self.save_traj(self.iteration_images, self.poses)
            tracking_fps = self.tracking_fps / self.iteration_images
            print(f"Average Tracking FPS: {tracking_fps:.2f}")
            self.tracking_avg_fps_shared[0] = tracking_fps
            self._write_downsample_summary()
            pbar.close()
            self.final_pose[:,:,:] = torch.tensor(np.array(self.poses)).float()
            self.end_of_dataset[0] = 1
            
            if len(self.gt_poses_matched) > 0:
                ate_rmse = self.evaluate_ate(self.gt_poses_matched, self.poses)
                self.tracking_ate_rmse_shared[0] = ate_rmse
                print(f"ATE RMSE: {ate_rmse:.2f} [m]")

        except Exception as e:
            print(f"!!!!!!!!!! TRACKER (Process-0) CRASHED !!!!!!!!!")
            print(f"!!!!!!!!!! ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print("--- TRACKER (Process-0): Function finished (exiting).")
    
    def save_traj(self, iter, poses):
        iter += 1
        traj = np.array([x[:3, 3] for x in poses])
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.title(f'Trajectory (Iter: {iter})')
        
        # Plot Estimated Trajectory
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label='estimated', linewidth=2)
        
        # Plot Ground Truth if available
        if len(self.gt_poses_matched) > 0:
            gt_poses_vis = np.array([x[:3, 3] for x in self.gt_poses_matched])
            # Limit GT size to match estimated length approximately or plot all
            # Usually plotting all GT is fine for context
            ax.plot(gt_poses_vis[:, 0], gt_poses_vis[:, 1], gt_poses_vis[:, 2], label='ground truth', linestyle='--')
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        
        # Set equal aspect ratio trick for 3D
        # Matplotlib 3D doesn't support 'axis equal' natively well, but we can approximate range
        all_pts = traj
        if len(self.gt_poses_matched) > 0:
            gt_poses_vis = np.array([x[:3, 3] for x in self.gt_poses_matched])
            all_pts = np.vstack((traj, gt_poses_vis))
            
        x_limits = [np.min(all_pts[:,0]), np.max(all_pts[:,0])]
        y_limits = [np.min(all_pts[:,1]), np.max(all_pts[:,1])]
        z_limits = [np.min(all_pts[:,2]), np.max(all_pts[:,2])]
        
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        
        max_range = max(x_range, y_range, z_range)
        
        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

        plt.savefig(f"{self.output_path}/traj_result.png")
        plt.close(fig)
    
    def _init_downsample_stats(self):
        self._downsample_frames = 0
        self._downsample_sum_before = 0
        self._downsample_sum_after = 0
        self._downsample_sum_trackable = 0

    def _accumulate_downsample_stats(self, num_before, num_after, num_trackable):
        self._downsample_frames += 1
        self._downsample_sum_before += int(num_before)
        self._downsample_sum_after += int(num_after)
        self._downsample_sum_trackable += int(num_trackable)
    
    def deskewing(
        self,
        raw_points: np.ndarray,
        ts,
        pose,
        ts_mid_pose=0.5,
    ):
        """
        Deskew a raw point array (N, 3/4) using per-point timestamps.
        """
        if ts is None:
            return raw_points

        pts_np = np.asarray(raw_points, dtype=np.float32)
        if pts_np.ndim != 2 or pts_np.shape[0] == 0 or pts_np.shape[1] < 3:
            return raw_points

        ts_np = np.asarray(ts).reshape(-1)
        if ts_np.shape[0] != pts_np.shape[0]:
            return raw_points

        finite_mask = np.isfinite(ts_np)
        if finite_mask.sum() < 2:
            return raw_points

        lo = float(np.min(ts_np[finite_mask]))
        hi = float(np.max(ts_np[finite_mask]))
        if not np.isfinite(hi - lo) or (hi - lo) <= 1e-12:
            return raw_points

        ts_t = torch.as_tensor((ts_np - lo) / (hi - lo), dtype=torch.float32)
        ts_t = ts_t - ts_mid_pose

        points_t = torch.from_numpy(pts_np[:, :3])
        pose_t = torch.as_tensor(pose, dtype=torch.float32)
        rotmat_slerp = roma.rotmat_slerp(
            torch.eye(3, dtype=torch.float32),
            pose_t[:3, :3],
            ts_t,
        )
        tran_lerp = ts_t[:, None] * pose_t[:3, 3]
        points_deskewd = (rotmat_slerp @ points_t.unsqueeze(-1)).squeeze(-1) + tran_lerp

        out = pts_np.copy()
        out[:, :3] = points_deskewd.cpu().numpy().astype(np.float32)
        return out

    def _write_downsample_summary(self):
        if self._downsample_frames == 0 or self._downsample_sum_before == 0:
            return

        avg_before = self._downsample_sum_before / self._downsample_frames
        avg_after = self._downsample_sum_after / self._downsample_frames
        avg_trackable = self._downsample_sum_trackable / self._downsample_frames
        ratio_after = self._downsample_sum_after / self._downsample_sum_before
        ratio_trackable = self._downsample_sum_trackable / self._downsample_sum_after if self._downsample_sum_after > 0 else 0.0

        summary_path = os.path.join(self.output_path, "downsample_summary.csv")
        with open(summary_path, "w") as f:
            f.write("voxel_size,frames,avg_before,avg_after,avg_trackable,ratio_after,ratio_trackable\n")
            f.write(f"{self.downsample_voxel_size},{self._downsample_frames},"
                    f"{avg_before:.2f},{avg_after:.2f},{avg_trackable:.2f},"
                    f"{ratio_after:.6f},{ratio_trackable:.6f}\n")
    
    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def quaternion_multiply(self, q1, Q2):
        x0, y0, z0, w0 = q1
        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                        w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                        w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                        w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T
    
    def eliminate_overlapped2(self, distances, threshold):
        return np.where(distances>threshold)
        
    def align(self, model, data):
        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3,-1))
        data_zerocentered = data - data.mean(1).reshape((3,-1))
        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U*S*Vh
        trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))
        model_aligned = rot * model + trans
        alignment_error = model_aligned - data
        trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]
        return rot, trans, trans_error

    def evaluate_ate(self, gt_traj, est_traj):
        n = min(len(gt_traj), len(est_traj))
        gt_traj_pts = np.array([gt_traj[idx][:3,3] for idx in range(n)]).T
        est_traj_pts = np.array([est_traj[idx][:3,3] for idx in range(n)]).T
        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)
        return trans_error.mean()
