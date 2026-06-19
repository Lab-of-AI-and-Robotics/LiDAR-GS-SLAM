import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import copy
import random
import sys
import cv2
import json
import numpy as np
import time
from datetime import datetime
import yaml
REPO_ROOT = os.path.dirname(__file__)
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "submodules", "MapClosures", "python"))
from arguments import SLAMParameters
from utils.loss_utils import l1_loss, ssim
from scene import GaussianModel
from gaussian_renderer import render,network_gui
from tqdm import tqdm
from scene.dataset_readers import get_dataset_reader
import matplotlib.pyplot as plt
import rerun as rr
# Additional import for densify
from utils.graphics_utils import depth_to_points, compute_depth_gradient, depth_to_normal, compute_normal_gradient
# Rotation matrix and quaternion utilities.
from utils.general_utils import create_rotation_matrix_from_direction_vector_batch, matrix_to_quaternion, inverse_sigmoid
# KNN distance computation.
from simple_knn._C import distCUDA2
from utils.sh_utils import RGB2SH
from datetime import datetime
import pygicp
from utils.PGO import PoseGraphManager
from map_closures.map_closures import MapClosures
from map_closures.config import MapClosuresConfig
from scene.dataset_readers import storePly
import threading, queue
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import subprocess
from omegaconf import OmegaConf

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        
class Mapper(SLAMParameters):
    def __init__(self, slam):   
        super().__init__()
        self.cfg = slam.cfg
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = float(slam.keyframe_th)
        self.trackable_opacity_th = slam.trackable_opacity_th
        self.trackable_conf_th = float(getattr(slam, "trackable_conf_th", 0.0))
        self.mapping_avg_fps_shared = slam.mapping_avg_fps_shared
        self.loop_cooldown_time = slam.loop_cooldown_time
        self.loop_next_allowed_frame = -1
        self.loop_cooldown_lock = threading.Lock()
        self.debug = True

        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        
        self.downsample_rate = slam.downsample_rate
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.densify_frequency = slam.densify_frequency

        # Densification parameters.
        self.densify_threshold_opacity = slam.densify_threshold_opacity
        self.densify_threshold_egeom = slam.densify_threshold_egeom
        self.densify_percentage = slam.densify_percentage
        self.pruning_min_opacity = slam.pruning_min_opacity
        self.pruning_min_scale = slam.pruning_min_scale
        self.opt_scaling_max = slam.opt_scaling_max
        self.opt_lambda_isotropy = slam.opt_lambda_isotropy
        self.opt_lambda_dist = slam.opt_lambda_dist
        self.decay_speed = slam.decay_speed
        self.use_densify = slam.use_densify
        self.densify_start_iteration = slam.densify_start_iteration
        self.densify_max_screen_size = float(getattr(slam, "densify_max_screen_size", 12.0))
        self.plane_xz_pass = bool(getattr(slam, "plane_xz_pass", True))
        self.plane_xz_pass_ratio = float(getattr(slam, "plane_xz_pass_ratio", 0.3))
        self.plane_xz_voxel_size = float(getattr(slam, "plane_xz_voxel_size", 0.8))
        self.plane_xz_max_candidates = int(getattr(slam, "plane_xz_max_candidates", 12000))
        self.plane_yz_pass = bool(getattr(slam, "plane_yz_pass", True))
        self.plane_yz_pass_ratio = float(getattr(slam, "plane_yz_pass_ratio", 0.2))
        self.plane_yz_voxel_size = float(getattr(slam, "plane_yz_voxel_size", 0.8))
        self.plane_yz_max_candidates = int(getattr(slam, "plane_yz_max_candidates", 8000))
        
        self.dataloader = None
        self.poses = []
        
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_mapping_keyframe = False
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
        self.loop_constraint_noise = slam.loop_constraint_noise
        self.knn_max_distance = slam.knn_max_distance
        
        self.gaussians = GaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.mapping_cams = []
        self.mapping_losses = []
        self.new_keyframes = []
        self.all_kf_poses = []
        self.all_kf_poses_idxs = []
        self.tracking_kf_idxs = []
        self.mapping_kf_idxs = []
        self.n_trackable_keyframes = slam.n_trackable_keyframes
        self.mapping_ate_rmse_shared = slam.mapping_ate_rmse_shared
        
        self.shared_cam = slam.shared_cam
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.final_gaussian_count_shared = slam.final_gaussian_count_shared
        self.gpu_avg_util_shared = slam.gpu_avg_util_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started

        self.collected_loops = []
        self.num_updated_loops = 0
        self.local_maps = []
        self.local_map_kf_idxs = []
        self.loop_max_corr = 30
        self.pgo_applied_to_final = False
        self.ate_rmse_shared = slam.ate_rmse_shared
        
        self.current_pose_shared = slam.current_pose_shared
        self.iter_shared = slam.iter_shared

        self.loop_task_q = queue.Queue(maxsize=8)
        self.loop_result_q = queue.Queue()
        cfg = MapClosuresConfig(
            density_map_resolution=0.5,
            density_threshold = 0.05,
            hamming_distance_threshold=50,
            inliers_threshold=5
        ) 
        self.loop_detector = MapClosures(cfg)
        self.loop_min_gap = 20  # Minimum keyframe gap.
        self.loop_voxel = cfg.density_map_resolution
        self.loop_worker = threading.Thread(target=self.loop_worker_fn, daemon=True)
        self.loop_worker.start()
        self.mapping_avg_fps_shared = slam.mapping_avg_fps_shared
        self.tracking_avg_fps_shared = slam.tracking_avg_fps_shared
        self.tracking_ate_rmse_shared = slam.tracking_ate_rmse_shared
        self.downsample_voxel_size = slam.downsample_voxel_size
        self.loop_overlap_th = slam.loop_overlap_th

    @torch.no_grad()
    def densify(self, viewpoint_cam):
        render_pkg = render(viewpoint_cam, self.gaussians, 0.0)

        mask_opacity = (render_pkg["rend_alpha"][0] <= self.densify_threshold_opacity)

        gt_depth = viewpoint_cam.original_depth_image[0]
        valid_pixels = (gt_depth > 0)

        densify_mask = (mask_opacity & valid_pixels)
        

        candidates = densify_mask.nonzero()
        no_samples = int(self.densify_percentage * candidates.shape[0])

        if no_samples < 2:
            return

        depth_gradient = compute_depth_gradient(viewpoint_cam.original_depth_image, 
                                                viewpoint_cam.original_depth_image>0)
        depth_gradient = depth_gradient / (depth_gradient.max() + 1e-6)
        gt_normal = viewpoint_cam.original_normal_image
        normal_grad = compute_normal_gradient(gt_normal)

        if depth_gradient[..., densify_mask].sum() <= 1e-5:
            return
        probs = depth_gradient[..., densify_mask].view(-1)
        sampled_indices = torch.multinomial(
            depth_gradient[..., densify_mask],
            no_samples
        )

        densify_mask_sampled = torch.zeros_like(densify_mask)
        densify_mask_sampled[candidates[sampled_indices, 0], candidates[sampled_indices, 1]] = 1.0

        points = depth_to_points(viewpoint_cam, viewpoint_cam.original_depth_image)
        points = points[..., densify_mask_sampled].T
        num_newpoints = points.shape[0]

        full_xyz = torch.cat((points, self.gaussians.get_xyz))

        dist2 = torch.clamp_max(
            torch.clamp_min(distCUDA2(full_xyz), 1e-7),
            self.opt_scaling_max**2
        )[:num_newpoints]
        base_scale = torch.sqrt(dist2)[..., None]
        scale_x = base_scale 
        scale_y = base_scale

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        y_coords = candidates[sampled_indices, 0]
        x_coords = candidates[sampled_indices, 1]
        normals_map = depth_to_normal(viewpoint_cam, viewpoint_cam.original_depth_image)
        normals = normals_map[..., densify_mask_sampled]
        new_normals = gt_normal[..., y_coords, x_coords].transpose(-1, -2).reshape(-1, 3)
        new_normals = new_normals.contiguous().float()


        c2w = torch.linalg.inv(viewpoint_cam.world_view_transform.T)
        R_c2w = c2w[:3, :3]

        normals = R_c2w @ normals

        n_R = create_rotation_matrix_from_direction_vector_batch(new_normals)
        rots = matrix_to_quaternion(n_R)

        opacities = inverse_sigmoid(0.9 * torch.ones((num_newpoints, 1), dtype=torch.float32, device="cuda"))

        colors = torch.ones((num_newpoints, 3), device="cuda")
        features_dc = RGB2SH(colors).unsqueeze(1)
        features_rest = torch.zeros((num_newpoints, self.gaussians._features_rest.shape[1], 3), device="cuda")

        new_trackable_mask = torch.zeros((num_newpoints), dtype=torch.bool, device="cuda")

        self.gaussians.densification_postfix(
            new_xyz=points, 
            new_features_dc=features_dc, 
            new_features_rest=features_rest, 
            new_opacities=opacities, 
            new_scaling=scales, 
            new_rotation=rots, 
            new_trackable_mask=new_trackable_mask
        )

        current_kf_idx = viewpoint_cam.cam_idx[0].item()
        new_keyframe_idx = torch.ones((num_newpoints, self.gaussians.keyframe_idx.shape[1]), device="cuda", dtype=torch.int32) * current_kf_idx
        self.gaussians.keyframe_idx = torch.concat([self.gaussians.keyframe_idx, new_keyframe_idx], dim=0)


    def run(self):
        self.dataloader = get_dataset_reader(self.cfg.data)
        self.poses = [self.dataloader.get_pose(0)]
        self.mapping()
        if self.end_of_dataset[0]:
            self.export_for_meshing()
    
    def mapping(self):
        t = torch.zeros((1,1)).float().cuda()
        if self.verbose:
            print("init start")
            network_gui.init("127.0.0.1", 6009)
        
        self.is_mapping_process_started[0] = 1
        
        while not self.is_tracking_keyframe_shared[0]:
            time.sleep(1e-5)
        
        newcam = copy.deepcopy(self.shared_cam)
        newcam.on_cuda()
  
        points, rots, scales, z_values, trackable_filter, control_score, phys_conf= self.shared_new_gaussians.get_values()
  
        self.gaussians.create_from_pcd2_tensor(points, rots, scales, z_values, trackable_filter, control_score, phys_conf, newcam.cam_idx[0])
        if self.debug:
            print("Create Initial Gaussian Done.")
        
        self.gaussians.spatial_lr_scale = self.scene_extent
        
        self.gaussians.training_setup(self)
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        
        if self.demo[0]:
            a = time.time()
            while (time.time()-a)<30.:
                print(f"Demo Waiting: {30.-(time.time()-a):.1f}s")
                self.run_viewer()
        self.demo[0] = 0
        gpu_util = 0.0
        
        self.tracking_kf_idxs.append(newcam.cam_idx[0])
        self.all_kf_poses_idxs.append(newcam.cam_idx[0])
        self.mapping_cams.append(newcam)
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.new_keyframes.append(len(self.mapping_cams)-1)
        R = newcam.R.detach().cpu().numpy()
        t = newcam.t.detach().cpu().numpy()
        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3,3] = t
        c2w = np.linalg.inv(w2c)
        self.all_kf_poses.append(c2w) 
        cam_idx = int(newcam.cam_idx[0])
        mask = (self.gaussians.keyframe_idx.squeeze(-1)==cam_idx)
        pts = self.gaussians.get_xyz[mask].detach().cpu()
        loop_count = 0
        prev_i = 0
        step_count = 0
        total_elapsed_time = 0.0
        while True:
            if self.end_of_dataset[0]: 
                mesh_pcd_path = os.path.join(self.output_path, "mesh_ready_pcd.ply")
                self.gaussians.save_pcd_for_mesh(mesh_pcd_path, opacity_threshold=0.8)
                self.final_gaussian_count_shared[0] = int(self.gaussians.get_xyz.shape[0])
                     
                self.gpu_avg_util_shared[0] = gpu_util / max(1, step_count)
                if len(self.collected_loops) > 0:
                    print("Applying Pose Graph Optimization to Final Trajectory...")
                    
                   

                    old_kf_poses, optimized_poses = self.pgo_update()
                    self.apply_pgo_to_final(old_kf_poses, optimized_poses)
                    ate_rmse = self.evaluate_ate(self.dataloader.gt_poses, self.final_pose.numpy())
                    self.mapping_ate_rmse_shared[0] = ate_rmse
                    print(f"ATE RMSE: {ate_rmse:.2f} [m]")
                    self.pgo_applied_to_final = True
                break
            start_time = time.perf_counter()
            self.consume_loop_results()
            if loop_count > 0:
                self.pgo_update()
                loop_count += (len(self.collected_loops) - loop_count)

            if self.is_tracking_keyframe_shared[0]:
                waiting_time = time.perf_counter() - start_time
                points, rots, scales, z_values, trackable_filter, control_score, phys_conf = self.shared_new_gaussians.get_values()
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
          
                self.gaussians.add_from_pcd2_tensor(points, rots, scales, z_values, trackable_filter,control_score, phys_conf, keyframe_idx = newcam.cam_idx[0])
               
                if len(self.tracking_kf_idxs) > self.n_trackable_keyframes:
                    cut_idx = self.tracking_kf_idxs[-self.n_trackable_keyframes-1].item()
                else:
                    cut_idx = -1

                target_setting_start = time.perf_counter()
                target_points, target_rots, target_scales, target_control  = self.gaussians.get_trackable_gaussians_tensor_trackble(self.trackable_opacity_th, cut_idx, self.trackable_conf_th)
                downsample_start_time = time.perf_counter()
                idx_np = self.voxel_downsample_indices(target_points.detach().cpu().numpy(),self.downsample_voxel_size)
                idx_t = torch.from_numpy(idx_np).to(target_points.device, dtype=torch.long)
                target_points = target_points.index_select(0, idx_t)
                target_rots = target_rots.index_select(0, idx_t)
                target_scales = target_scales.index_select(0, idx_t)
                target_control = target_control.index_select(0, idx_t)
                downsample_duration = time.perf_counter() - downsample_start_time
              
                
                self.shared_target_gaussians.input_values(target_points, target_rots, target_scales, target_control)
                self.target_gaussians_ready[0] = 1
                target_setting_duration = time.perf_counter() - target_setting_start

                
            
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.tracking_kf_idxs.append(newcam.cam_idx[0])
                self.all_kf_poses_idxs.append(newcam.cam_idx[0])
                self.is_tracking_keyframe_shared[0]=0
                R = newcam.R.detach().cpu().numpy()
                t = newcam.t.detach().cpu().numpy()
                w2c = np.eye(4)
                w2c[:3,:3] = R
                w2c[:3,3] = t
                c2w = np.linalg.inv(w2c)
                self.all_kf_poses.append(c2w) 
                mask = (self.gaussians.keyframe_idx.squeeze(-1)==newcam.cam_idx[0])
                pts = self.gaussians.get_xyz[mask].detach().cpu()
                self.save_local_map(pts,w2c,newcam.cam_idx[0])
                query_id = len(self.local_maps)-1
                self.try_loop_closure(query_id)

            elif self.is_mapping_keyframe_shared[0]:
                points, rots, scales, z_values, _ ,control_score, phys_conf= self.shared_new_gaussians.get_values()
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.gaussians.add_from_pcd2_tensor(points, rots, scales, z_values, [], control_score, phys_conf, keyframe_idx = newcam.cam_idx[0])
                if len(self.tracking_kf_idxs) > self.n_trackable_keyframes:
                    cut_idx = self.tracking_kf_idxs[-self.n_trackable_keyframes].item()
                else:
                    cut_idx = 0

               
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.mapping_kf_idxs.append(newcam.cam_idx[0])
                self.all_kf_poses_idxs.append(newcam.cam_idx[0])
                self.is_mapping_keyframe_shared[0] = 0
                R = newcam.R.detach().cpu().numpy()
                t = newcam.t.detach().cpu().numpy()
                w2c = np.eye(4)
                w2c[:3,:3] = R
                w2c[:3,3] = t
                c2w = np.linalg.inv(w2c)
                self.all_kf_poses.append(c2w) 
                mask = (self.gaussians.keyframe_idx.squeeze(-1)==newcam.cam_idx[0])
                pts = self.gaussians.get_xyz[mask].detach().cpu()
                self.save_local_map(pts,w2c,newcam.cam_idx[0])
                query_id = len(self.local_maps)-1
                self.try_loop_closure(query_id)

                
            if len(self.mapping_cams)>0:
                if len(self.new_keyframes) > 0:
                    train_idx = self.new_keyframes.pop(0)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    new_keyframe = True


                else:
                    train_idx = random.choice(range(len(self.mapping_cams)))
                    viewpoint_cam = self.mapping_cams[train_idx]
                self.training = True
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                xyz = self.gaussians.get_xyz
                R_cam = viewpoint_cam.R.to(xyz.device)
                t_cam = viewpoint_cam.t.to(xyz.device)
                pts_cam = xyz @ R_cam.transpose(0, 1) + t_cam
                z = pts_cam[:, 2]
                fx = viewpoint_cam.fx.to(xyz.device)
                fy = viewpoint_cam.fy.to(xyz.device)
                cx = viewpoint_cam.cx.to(xyz.device)
                cy = viewpoint_cam.cy.to(xyz.device)
                u = fx * (pts_cam[:, 0] / z) + cx
                v = fy * (pts_cam[:, 1] / z) + cy
                W = float(viewpoint_cam.image_width[0].item())
                H = float(viewpoint_cam.image_height[0].item())
                dist = torch.norm(pts_cam, dim=-1)
                vis_mask = (z > 0) & (z < self.depth_trunc) 
                active_idx = torch.where(vis_mask)[0]
                if active_idx.numel() == 0:
                    active_idx = torch.arange(xyz.shape[0], device=xyz.device)
                
                
                render_pkg = render(viewpoint_cam, self.gaussians, 0.5, gaussian_indices=active_idx)
                est_alpha = render_pkg["rend_alpha"]
                est_depth = render_pkg["surf_depth"]
                est_normal = render_pkg["rend_normal"]
                surf_normal = render_pkg["surf_normal"]

                rend_dist = render_pkg["rend_dist"]

                gt_depth = viewpoint_cam.original_depth_image.cuda()

                gt_normal = viewpoint_cam.original_normal_image.cuda()
                valid_mask = (gt_depth > 0).float()


                _ , ssim_val = ssim(est_depth.unsqueeze(0), gt_depth.unsqueeze(0))
                dssim_loss = (1.0 - ssim_val) * self.lambda_dssim
                valid_mask_gicp_normal = (gt_normal != 0).float()
                valid_mask_gicp_normal *= valid_mask
                
                if est_depth.dim() == 2: est_depth = est_depth.unsqueeze(0)
                if est_alpha.dim() == 2: est_alpha = est_alpha.unsqueeze(0)
                if valid_mask.dim() == 2: valid_mask = valid_mask.unsqueeze(0)


                geom_l1 = torch.abs(valid_mask * (est_depth - gt_depth)).mean()

                normal_loss_gradient = (1 - (
                    est_normal[..., valid_mask[0] == 1.0] * surf_normal[..., valid_mask[0] == 1.0]
                ).sum(dim=0)).mean()

                # GICP-based normal loss (from covariance)
                normal_loss_gicp = (1 - (
                    est_normal[..., valid_mask_gicp_normal[0] == 1.0] * gt_normal[..., valid_mask_gicp_normal[0] == 1.0]
                ).sum(dim=0)).mean()

                normal_loss_gicp *= 0.05
                if self.train_iter >= self.densify_start_iteration :
                    normal_loss_gradient *= 0.01
                else :
                    normal_loss_gradient = 0.0
                normal_loss = normal_loss_gicp + normal_loss_gradient

                # Alpha loss
                alpha_loss = torch.nn.functional.binary_cross_entropy(
                    est_alpha[..., valid_mask[0] == 1.0],
                    valid_mask[..., valid_mask[0] == 1.0],
                    reduction="mean"
                )
                alpha_loss *= self.opt_lambda_alpha

                scales_max = self.gaussians.get_scaling.max(dim=1).values
                gaussian_mean = self.gaussians.get_xyz
                opt_scaling_max = self.opt_scaling_max
                reg_scales_tensor = scales_max[scales_max >= opt_scaling_max] - opt_scaling_max
                reg_scales = (self.opt_scaling_max_penalty * reg_scales_tensor).sum()

                loss_total = geom_l1 + alpha_loss  + reg_scales + normal_loss

                loss_total.backward()
                self.gaussians.optimizer.step()

                if (self.use_densify and
                    self.train_iter >= self.densify_start_iteration and
                    self.train_iter % self.densify_frequency == 0):
                    max_screen_size = self.densify_max_screen_size if self.densify_max_screen_size > 0 else None
                    with torch.no_grad():
                        self.gaussians.densify_and_prune(
                            max_grad=self.densify_grad_threshold,
                            min_opacity=self.pruning_min_opacity,
                            extent=None,
                            max_screen_size=max_screen_size
                    )
                
                plane_manage_frequency = max(1, int(self.densify_frequency) * 1.5)
                plane_manage_frequency = 300
                plane_manage_start_iteration = 300 
                if (self.use_densify and
                    self.train_iter >= plane_manage_start_iteration and
                    self.train_iter % plane_manage_frequency == 0):
                    plane_manage_time = time.perf_counter()
                    with torch.no_grad():
                        plane_stats = self.gaussians.plane_surface_manage(
                            min_opacity=self.pruning_min_opacity,
                            enable_xz_pass=self.plane_xz_pass,
                            xz_pass_ratio=self.plane_xz_pass_ratio,
                            xz_voxel_size=self.plane_xz_voxel_size,
                            xz_max_candidates=self.plane_xz_max_candidates,
                            enable_yz_pass=self.plane_yz_pass,
                            yz_pass_ratio=self.plane_yz_pass_ratio,
                            yz_voxel_size=self.plane_yz_voxel_size,
                            yz_max_candidates=self.plane_yz_max_candidates,
                        )
                        plane_manage_duration = time.perf_counter() - plane_manage_time
             
                elapsed = time.perf_counter()-start_time
                gpu_util += self.read_gpu_util()
                self.train_iter += 1
                total_elapsed_time += elapsed
                if total_elapsed_time > 0:
                    avg_fps = self.train_iter / total_elapsed_time
                    self.mapping_avg_fps_shared[0] = avg_fps
                
                torch.cuda.empty_cache()
    def read_gpu_util(self):
        out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
         ).decode().strip().splitlines()
        vals = [float(x) for x in out if x.strip()]
        return sum(vals) / len(vals) if vals else 0.0

    def export_for_meshing(self):
        """
        Export the final Gaussian map and a set of poses for meshing / evaluation
        in Splat-LOAM format.
        """

        try:
            os.makedirs(self.output_path, exist_ok=True)
        except Exception:
            pass
        
        # Create Splat-LOAM compatible directory structure
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_folder = os.path.join(self.output_path, "results", date_str)
        models_folder = os.path.join(result_folder, "models")
        os.makedirs(models_folder, exist_ok=True)
        
        print(f"[Mapper] Exporting results to {result_folder}")

        # 1) Save Gaussians to PLY (Model 0)
        ply_path = os.path.join(models_folder, "0000.ply")
        try:
            self.gaussians.save_ply(ply_path)
            print(f"[Mapper] Saved gaussians to {ply_path}")
        except Exception as e:
            print(f"[Mapper] Failed to save gaussians .ply: {e}")
            return None

        # 2) Build graph.yaml and frames
        frames_list = []
        frame_ids = []
        
        # Use mapping_cams to reconstruct trajectory and camera info
        if not self.mapping_cams:
            print("[Mapper] No mapping cameras found. Skipping graph export.")
            return None

        for i, cam in enumerate(self.mapping_cams):
            # cam.world_view_transform is w2c (likely transposed from Camera init)
            # We want c2w (model_T_frame)
            
            # Check tensor device and convert to numpy
            w2c_T = cam.world_view_transform.detach().cpu().numpy()
            w2c = w2c_T.T # Transpose back to standard [R t; 0 1]
            
            try:
                c2w = np.linalg.inv(w2c)
            except np.linalg.LinAlgError:
                print(f"[Mapper] Singular matrix for cam {i}, skipping.")
                continue
                
            # Flatten 3x4 (row-major)
            mTf = c2w[:3, :].flatten().tolist()
            
            # Projection parameters
            def get_val(x):
                if isinstance(x, torch.Tensor):
                    return float(x.item())
                return float(x)

            proj = [get_val(cam.fx), get_val(cam.fy), get_val(cam.cx), get_val(cam.cy)]
            
            frame_entry = {
                "id": i,
                "timestamp": float(i), # Use index as dummy timestamp
                "model_T_frame": mTf,
                "projmatrix": proj,
                "model_id": 0
            }
            frames_list.append(frame_entry)
            frame_ids.append(i)

        model_entry = {
            "id": 0,
            "world_T_model": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # Identity
            "filename": "models/0000.ply",
            "frame_ids": frame_ids
        }
        
        graph_data = {
            "models": [model_entry],
            "frames": frames_list
        }
        
        graph_path = os.path.join(result_folder, "graph.yaml")
        try:
            with open(graph_path, 'w') as f:
                yaml.dump(graph_data, f)
            print(f"[Mapper] Saved graph to {graph_path}")
        except Exception as e:
            print(f"[Mapper] Failed to save graph.yaml: {e}")
            return None

        # 3) Save minimal cfg.yaml
        cfg_data = {
            "preprocessing": {
                "image_width": int(self.W),
                "image_height": int(self.H)
            },
            "device": "cuda",
             "opt": {
                "depth_ratio": 0.0
            }
        }
        cfg_path = os.path.join(result_folder, "cfg.yaml")
        try:
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg_data, f)
            print(f"[Mapper] Saved config to {cfg_path}")
        except Exception as e:
            print(f"[Mapper] Failed to save cfg.yaml: {e}")
            return None

        metrics = self._collect_experiment_metrics(result_folder)
        metrics_path = os.path.join(result_folder, "metrics.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[Mapper] Saved metrics to {metrics_path}")
        except Exception as e:
            print(f"[Mapper] Failed to save metrics.json: {e}")
            return None

        try:
            self._save_est_traj_for_evo(result_folder)
        except Exception as e:
            print(f"[Mapper] Failed to save trajectory txt files: {e}")
            return None

        summary_path = os.path.join(result_folder, "summary.md")
        try:
            self._write_experiment_summary(summary_path, metrics)
            print(f"[Mapper] Saved summary to {summary_path}")
        except Exception as e:
            print(f"[Mapper] Failed to save summary.md: {e}")
            return None

        print(f"Results ready at {result_folder}")
        print(f"You can now run: python3 Splat-LOAM/run.py mesh {result_folder}")
        return result_folder

    def _collect_experiment_metrics(self, result_folder):
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        data_cfg = cfg_dict.get("data", {}) if isinstance(cfg_dict, dict) else {}
        slam_cfg = cfg_dict.get("slam", {}) if isinstance(cfg_dict, dict) else {}
        final_poses_np = self.final_pose.detach().cpu().numpy()
        num_frames = int(
            sum(np.isfinite(pose).all() and not np.allclose(pose, 0.0) for pose in final_poses_np)
        )
        if num_frames == 0:
            num_frames = len(self.poses)
        num_keyframes = len(self.mapping_cams)
        final_gaussians = int(self.final_gaussian_count_shared[0].item())
        if final_gaussians <= 0:
            final_gaussians = int(self.gaussians.get_xyz.shape[0])

        return {
            "status": "completed",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "result_folder": result_folder,
            "dataset_type": str(data_cfg.get("dataset_type", "")),
            "dataset_path": str(data_cfg.get("dataset_path", "")),
            "num_frames": num_frames,
            "num_keyframes": num_keyframes,
            "num_loop_closures": len(self.collected_loops),
            "pgo_applied_to_final": bool(self.pgo_applied_to_final),
            "tracking_avg_fps": float(self.tracking_avg_fps_shared[0].item()),
            "mapping_avg_fps": float(self.mapping_avg_fps_shared[0].item()),
            "tracking_ate_rmse": float(self.tracking_ate_rmse_shared[0].item()),
            "mapping_ate_rmse": float(self.mapping_ate_rmse_shared[0].item()),
            "final_gaussian_count": final_gaussians,
            "gpu_avg_util": float(self.gpu_avg_util_shared[0].item()),
            "downsample_rate": int(slam_cfg.get("downsample_rate", self.downsample_rate)),
            "downsample_voxel_size": float(slam_cfg.get("downsample_voxel_size", self.downsample_voxel_size)),
            "keyframe_freq": int(slam_cfg.get("keyframe_freq", self.keyframe_freq)),
            "use_densify": bool(slam_cfg.get("use_densify", self.use_densify)),
            "artifacts": {
                "gaussians": "models/0000.ply",
                "graph": "graph.yaml",
                "mesh_config": "cfg.yaml",
                "summary": "summary.md",
                "trajectory_tum": "est_traj_tum.txt",
                "trajectory_kitti": "est_traj_kitti.txt",
                "trajectory_points": "est_traj_pts.txt",
            },
        }

    def _save_est_traj_for_evo(self, result_folder):
        poses = self.final_pose.detach().cpu().numpy()
        valid = [pose for pose in poses if np.isfinite(pose).all() and not np.allclose(pose, 0.0)]
        if not valid:
            valid = self.poses

        tum_path = os.path.join(result_folder, "est_traj_tum.txt")
        kitti_path = os.path.join(result_folder, "est_traj_kitti.txt")
        pts_path = os.path.join(result_folder, "est_traj_pts.txt")

        with open(tum_path, "w", encoding="utf-8") as f_tum, \
             open(kitti_path, "w", encoding="utf-8") as f_kitti, \
             open(pts_path, "w", encoding="utf-8") as f_pts:
            for idx, pose in enumerate(valid):
                pose = np.asarray(pose, dtype=np.float64)
                timestamp = self._get_export_timestamp(idx)
                t = pose[:3, 3]
                qx, qy, qz, qw = R.from_matrix(pose[:3, :3]).as_quat()
                f_tum.write(
                    f"{timestamp:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
                )
                f_kitti.write(" ".join(f"{x:.9f}" for x in pose[:3, :].reshape(-1)) + "\n")
                f_pts.write(f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f}\n")

        print(f"[Mapper] Saved trajectories to {result_folder}")

    def _get_export_timestamp(self, idx):
        try:
            ts = self.dataloader.get_timestamp(idx)
            return float(ts)
        except Exception:
            return float(idx)

    def _write_experiment_summary(self, summary_path, metrics):
        lines = [
            "# Experiment Summary",
            "",
            f"- Status: {metrics['status']}",
            f"- Created: {metrics['created_at']}",
            f"- Dataset: {metrics['dataset_type']}",
            f"- Dataset path: `{metrics['dataset_path']}`",
            f"- Frames: {metrics['num_frames']}",
            f"- Keyframes: {metrics['num_keyframes']}",
            f"- Loop closures: {metrics['num_loop_closures']}",
            f"- Final Gaussians: {metrics['final_gaussian_count']}",
            f"- Tracking FPS: {metrics['tracking_avg_fps']:.3f}",
            f"- Mapping FPS: {metrics['mapping_avg_fps']:.3f}",
            f"- Tracking ATE RMSE: {metrics['tracking_ate_rmse']:.6f}",
            f"- Mapping ATE RMSE: {metrics['mapping_ate_rmse']:.6f}",
            f"- GPU avg util: {metrics['gpu_avg_util']:.3f}",
            "",
            "## Artifacts",
            "",
        ]
        for _, rel_path in metrics["artifacts"].items():
            lines.append(f"- `{rel_path}`")
        lines.extend([
            "",
            "## Reconstruction",
            "",
            "```bash",
            f"python3 Splat-LOAM/run.py mesh {metrics['result_folder']}",
            "```",
            "",
        ])
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

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
                    net_image0 = render(custom_cam, self.gaussians, 0.5)["surf_depth"]
                    net_image = net_image0.repeat(3,1,1)
                    net_image_bytes = memoryview(((net_image * 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()))

                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
    def save_local_map(self, target_points, w2c, cam_idx):
        if target_points.numel() == 0:
            pts_local = np.empty((0,3),dtype=np.float32)
        else:
            pts = target_points.numpy().astype(np.float32)
            pts_local = (w2c[:3,:3]@pts.T).T + w2c[:3,3]
        self.local_maps.append([pts_local])
        self.local_map_kf_idxs.append(int(cam_idx))

    def voxel_downsample_indices(self, points, voxel_size):
        # points: (N,3)
        if voxel_size <= 0:
            return np.arange(points.shape[0])
        min_pt = points.min(axis=0)
        
        vox = ((points - min_pt) / voxel_size).astype(np.int32)
        order = np.lexsort((vox[:, 2], vox[:, 1], vox[:, 0]))
        vox_sorted = vox[order]

        mask = np.ones(len(vox_sorted), dtype=bool)
        mask[1:] = np.any(vox_sorted[1:] != vox_sorted[:-1], axis=1)
        return order[mask]

       

    def try_loop_closure(self, query_id:int) -> bool:
        current_frame = query_id
        
        if query_id < 0 or query_id >= len(self.local_maps):
            return False
        
        with self.loop_cooldown_lock:
            if current_frame < self.loop_next_allowed_frame:
                return False
        
        query_points = self.local_maps[query_id][0]
        if query_points.shape[0] == 0:
            return False

        try:
            self.loop_task_q.put_nowait((query_id, query_points.copy()))
        except queue.Full:
            return False
        return True
    def loop_worker_fn(self):
        while True:
            task = self.loop_task_q.get()
            if task is None:
                break
            
            try:
                query_id, query_points = task
                current_frame = query_id
                with self.loop_cooldown_lock:
                    if current_frame < self.loop_next_allowed_frame:
                        continue
                closure = self.loop_detector.get_best_closure(query_id, query_points)
        
                if closure.number_of_inliers < self.loop_detector._config.inliers_threshold:
                    continue
                
                src_pts = self.local_maps[closure.source_id][0]
                tgt_pts = self.local_maps[closure.target_id][0]

                ok, T = self.validate_closure(src_pts, tgt_pts, np.asarray(closure.pose))
                if not ok:
                    continue

                with self.loop_cooldown_lock:
                    self.loop_next_allowed_frame = max(
                        self.loop_next_allowed_frame,
                        current_frame + self.loop_cooldown_time
                    )

                self.loop_result_q.put((closure.source_id, closure.target_id, T))
            except Exception as e:
                print(f"[Loop Worker] Exception: {e}")
                continue

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

        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error
    
    def save_kf_traj(self, poses, save_path, gt_kf=None):
        pts = np.array([p[:3, 3] for p in poses])
        plt.clf()
        plt.plot(pts[:, 0], pts[:, 2], label="est", linewidth=3)
        if gt_kf is not None:
            gt_pts = np.array([p[:3, 3] for p in gt_kf])
            plt.plot(gt_pts[:, 0], gt_pts[:, 2], label="gt")
        plt.axis("equal")
        plt.legend()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    
    def apply_pgo_to_final(self, old_kf_poses, new_kf_poses):
        cam_to_pose_idx = {int(cam_idx): i for i, cam_idx in enumerate(self.all_kf_poses_idxs)}

        final_pose_np = self.final_pose.numpy()
        n_frames = final_pose_np.shape[0]

        kf_list = [int(x) for x in self.tracking_kf_idxs]
        kf_list = sorted(kf_list)
        if len(kf_list) == 0:
            return
        if len(kf_list) == 1:
            final_pose_np[:] = new_kf_poses[cam_to_pose_idx[kf_list[0]]]
            return

        kf_poses = [new_kf_poses[cam_to_pose_idx[k]] for k in kf_list]

        # Fill frames before the first keyframe and after the last keyframe.
        final_pose_np[:kf_list[0]] = kf_poses[0]
        final_pose_np[kf_list[-1]:] = kf_poses[-1]

        for i in range(len(kf_list) - 1):
            f0, f1 = kf_list[i], kf_list[i + 1]
            if f1 <= f0:
                continue

            T0 = kf_poses[i]
            T1 = kf_poses[i + 1]

            R01 = R.from_matrix([T0[:3, :3], T1[:3, :3]])
            slerp = Slerp([0.0, 1.0], R01)

            ts = np.linspace(0.0, 1.0, f1 - f0, endpoint=False)
            Rs = slerp(ts).as_matrix()

            t0 = T0[:3, 3]
            t1 = T1[:3, 3]
            trans = (1.0 - ts)[:, None] * t0 + ts[:, None] * t1

            for j, frame in enumerate(range(f0, f1)):
                final_pose_np[frame, :3, :3] = Rs[j]
                final_pose_np[frame, :3, 3] = trans[j]
                final_pose_np[frame, 3, :] = [0.0, 0.0, 0.0, 1.0]

    def evaluate_ate(self, gt_traj, est_traj):
        n = min(len(gt_traj), len(est_traj))
        gt_traj_pts = np.array([gt_traj[idx][:3,3] for idx in range(n)]).T
        est_traj_pts = np.array([est_traj[idx][:3,3] for idx in range(n)]).T
        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)
        return trans_error.mean()


    def pgo_update(self):
        old_kf_poses = copy.deepcopy(self.all_kf_poses)
        PGM = PoseGraphManager(self.loop_constraint_noise)
        
        if self.num_updated_loops >= len(self.collected_loops) and not self.end_of_dataset[0]:
            return False
        
        for i, kf_pose in enumerate(self.all_kf_poses):
            if i == 0:
                PGM.addPriorFactor(0, kf_pose)
            else:
                prev_kf_pose = self.all_kf_poses[i-1]
                relative_pose = np.linalg.inv(prev_kf_pose) @ kf_pose
                PGM.addOdometryFactor(i-1, i, relative_T=relative_pose, initial_T=kf_pose)
        
        for source_idx, target_idx, relative_pose in self.collected_loops:
            PGM.addLoopFactor(source_idx, target_idx, relative_T=relative_pose)

        PGM.optimizePoseGraph()
        
        optimized_poses = PGM.getValues(len(self.all_kf_poses))
                   
        for i, kf_pose in enumerate(self.all_kf_poses):

            delta = optimized_poses[i]@np.linalg.inv(kf_pose)
            cam_idx = int(self.all_kf_poses_idxs[i])
            self.gaussians.transform_gaussians(cam_idx, delta)

            if i < len(self.mapping_cams):
                c2w = optimized_poses[i]
                w2c = np.linalg.inv(c2w)
                cam = self.mapping_cams[i]
                cam.R = torch.as_tensor(w2c[:3, :3], dtype=cam.R.dtype, device=cam.R.device)
                cam.t = torch.as_tensor(w2c[:3, 3], dtype=cam.t.dtype, device=cam.t.device)
                cam.update_matrix()

            self.all_kf_poses[i] = optimized_poses[i]
        
        kf_ids = [int(x) for x in self.tracking_kf_idxs]
        gt_kf_all = [self.dataloader.get_pose(k) for k in kf_ids]
        gt_kf = [pose for pose in gt_kf_all if pose is not None]

        self.save_kf_traj(old_kf_poses, os.path.join(self.output_path, "traj_kf_before.png"), gt_kf)
        self.save_kf_traj(optimized_poses, os.path.join(self.output_path, "traj_kf_after.png"), gt_kf)
        self.kf_poses_before_pgo = []
        self.kf_poses_after_pgo = []
        before_pgo_vis = []
        after_pgo_vis = []
        for i, kf_pose in enumerate(self.all_kf_poses):
            
            self.kf_poses_before_pgo.append(kf_pose)
            self.kf_poses_after_pgo.append(optimized_poses[i])
            before_pgo_vis.append(kf_pose[:3,3])
            after_pgo_vis.append(optimized_poses[i][:3,3])

        self.num_updated_loops = len(self.collected_loops)

        return old_kf_poses, optimized_poses 


    def validate_closure(self, source_pts, target_pts, init_T):
        src = pygicp.downsample(source_pts, self.loop_voxel)
        tgt = pygicp.downsample(target_pts, self.loop_voxel)
        if src.shape[0] == 0 or tgt.shape[0] == 0:
            return False, init_T
        
        reg = pygicp.FastGICP()
        reg.set_max_correspondence_distance(self.loop_max_corr)
        reg.set_max_knn_distance(self.knn_max_distance)
        reg.set_num_threads(1)

        reg.set_input_target(tgt)
        reg.calculate_target_covariance()
        reg.set_input_source(src)
        reg.calculate_source_covariance()
        
        reg.set_target_control_score((5/9)*np.ones(tgt.shape[0], dtype=np.float32))
        T = reg.align(init_T)  # source -> target
        fitness = reg.get_fitness_score(self.loop_max_corr)
        ok = fitness < self.loop_overlap_th
        return ok, T
    
    def consume_loop_results(self):
        while not self.loop_result_q.empty():
            src_id, tgt_id, T = self.loop_result_q.get()
            src_cam = int(self.local_map_kf_idxs[src_id])
            tgt_cam = int(self.local_map_kf_idxs[tgt_id])

            matches = [i for i, x in enumerate(self.all_kf_poses_idxs) if int(x) == src_cam]
            if len(matches) != 1:
                print(f"[Loop IDX] src_cam={src_cam} matches={matches}")


            try:
                src_pose_idx = self.all_kf_poses_idxs.index(src_cam)
                tgt_pose_idx = self.all_kf_poses_idxs.index(tgt_cam)
            except ValueError:
                continue
           
            self.collected_loops.append((src_pose_idx, tgt_pose_idx, np.linalg.inv(T)))
