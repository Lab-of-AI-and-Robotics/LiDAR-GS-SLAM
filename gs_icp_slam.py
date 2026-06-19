import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import sys
import cv2
import numpy as np
import open3d as o3d
import time
from pathlib import Path
sys.path.append(os.path.dirname(__file__))
from argparse import ArgumentParser
from arguments import SLAMParameters
from utils.graphics_utils import focal2fov, fov2focal
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints
from gaussian_renderer import render, network_gui
from mp_Tracker import Tracker
from mp_Mapper import Mapper
import json
from scene.dataset_readers import get_dataset_reader
from utils.config_utils import load_configuration, DatasetType

torch.multiprocessing.set_sharing_strategy('file_system')

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        
class GS_ICP_SLAM(SLAMParameters):
    def __init__(self, args, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_path = cfg.data.dataset_path
        if args.output_path: cfg.output_path = args.output_path
        self.output_path = cfg.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = cfg.verbose
        self.tracking_ate_rmse_shared = torch.zeros((1)).float()
        self.tracking_ate_rmse_shared.share_memory_()

        self.mapping_ate_rmse_shared = torch.zeros((1)).float()
        self.mapping_ate_rmse_shared.share_memory_()
        self.keyframe_th = float(cfg.slam.keyframe_th)
        self.knn_max_distance = float(cfg.slam.knn_maxd)
        self.overlapped_th = float(cfg.slam.overlapped_th)
        self.max_correspondence_distance = float(cfg.slam.max_correspondence_distance)
        self.trackable_opacity_th = float(cfg.slam.trackable_opacity_th)
        self.overlapped_th2 = float(cfg.slam.overlapped_th2)
        self.downsample_rate = int(cfg.slam.downsample_rate)
        self.keyframe_freq = int(cfg.slam.keyframe_freq)
        self.n_trackable_keyframes = int(cfg.slam.n_trackable_keyframes)
        self.test = cfg.test
        self.loop_constraint_noise = cfg.slam.loop_constraint_noise
        self.downsample_voxel_size = cfg.slam.downsample_voxel_size
        self.loop_overlap_th = float(cfg.slam.loop_overlap_th) if hasattr(cfg.slam, 'loop_overlap_th') else 0.01
        self.loop_cooldown_time = float(cfg.slam.loop_cooldown_time) if hasattr(cfg.slam, 'loop_cooldown_time') else 5.0
        self.use_dynamic_fov = cfg.slam.use_dynamic_fov

        # densify related parameters
        self.densify_threshold_opacity = float(cfg.slam.densify_threshold_opacity)
        self.densify_threshold_egeom = float(cfg.slam.densify_threshold_egeom) if hasattr(cfg.slam, 'densify_threshold_egeom') else 0.01
        self.densify_percentage = float(cfg.slam.densify_percentage)
        self.opt_scaling_max = float(cfg.slam.opt_scaling_max)
        self.opt_lambda_isotropy = float(cfg.slam.opt_lambda_isotropy) if hasattr(cfg.slam, 'opt_lambda_isotropy') else 0.05
        self.opt_lambda_dist = float(cfg.slam.opt_lambda_dist)
        self.decay_speed = float(cfg.slam.decay_speed)
        self.use_densify = cfg.slam.use_densify
        self.densify_start_iteration = int(cfg.slam.densify_start_iteration)
        self.rerun_viewer = args.rerun_viewer
        self.densify_frequency = args.densify_frequency if args.densify_frequency is not None else (cfg.slam.densify_frequency if hasattr(cfg.slam, 'densify_frequency') else 50)
        self.trackable_conf_th = float(cfg.slam.trackable_conf_th) if hasattr(cfg.slam, 'trackable_conf_th') else 0.0

        # pruning related parameters
        self.pruning_min_opacity = float(cfg.slam.pruning_min_opacity)
        self.pruning_min_scale = float(cfg.slam.pruning_min_scale)
        
        # LiDAR Parameters
        self.W = cfg.slam.W
        self.H = cfg.slam.H     
        self.fx = fov2focal(np.deg2rad(cfg.slam.fov_x_deg), self.W)
        self.fy = fov2focal(np.deg2rad(cfg.slam.fov_y_deg), self.H)
        self.cx = self.W/2
        self.cy = self.H/2
        self.depth_scale = cfg.slam.depth_scale
        self.depth_trunc = cfg.slam.depth_trunc
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Create Dataset Reader ONLY to get initial info
        # We do NOT store self.dataloader as member to avoid pickling open files
        temp_loader = get_dataset_reader(self.cfg.data)
        
        # Get first frame for initialization
        test_points, _, _, _ = temp_loader[0]

        # Get size of final poses
        num_final_poses = len(temp_loader)

        # Use dimensions from configuration
        H, W = self.H, self.W
        depth = np.zeros((H, W), dtype=np.float32)
        
        # Shared objects
        # Sensor info
        self.shared_cam = SharedCam(FoVx=focal2fov(self.fx, W), FoVy=focal2fov(self.fy, H),
                                    depth_image=depth,
                                    cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)

        # Initialize with a sufficiently large buffer size (e.g., 2,000,000) to handle varying point cloud sizes
        self.shared_new_gaussians = SharedGaussians(2000000)
        # Mapper -> Tracker
        self.shared_target_gaussians = SharedTargetPoints(25000000)

        # Trigger
        self.end_of_dataset = torch.zeros((1)).int()
        self.is_tracking_keyframe_shared = torch.zeros((1)).int()
        self.is_mapping_keyframe_shared = torch.zeros((1)).int()
        self.target_gaussians_ready = torch.zeros((1)).int()
        self.new_points_ready = torch.zeros((1)).int()
        self.is_mapping_process_started = torch.zeros((1)).int()

        self.final_pose = torch.zeros((num_final_poses,4,4)).float()
        self.demo = torch.zeros((1)).int()
        
        self.shared_cam.share_memory()
        self.shared_new_gaussians.share_memory()
        self.shared_target_gaussians.share_memory()
        self.end_of_dataset.share_memory_()
        self.is_tracking_keyframe_shared.share_memory_()
        self.is_mapping_keyframe_shared.share_memory_()
        self.target_gaussians_ready.share_memory_()
        self.new_points_ready.share_memory_()
        self.final_pose.share_memory_()
        self.demo.share_memory_()
        self.is_mapping_process_started.share_memory_()
        
        self.current_pose_shared = torch.eye(4).float().cuda()
        self.current_pose_shared.share_memory_()
        self.iter_shared = torch.zeros((1)).int()
        self.iter_shared.share_memory_()
        self.ate_rmse_shared = torch.zeros((1)).float()
        self.ate_rmse_shared.share_memory_()
        self.tracking_avg_fps_shared = torch.zeros((1)).float()
        self.tracking_avg_fps_shared.share_memory_()
        self.mapping_avg_fps_shared = torch.zeros((1)).float()
        self.mapping_avg_fps_shared.share_memory_()
        self.final_gaussian_count_shared = torch.zeros((1)).int()
        self.final_gaussian_count_shared.share_memory_()
        self.gpu_avg_util_shared = torch.zeros((1)).float()
        self.gpu_avg_util_shared.float().share_memory_()
        
        
        
        self.demo[0] = cfg.demo


    def tracking(self, rank):
        self.tracker = Tracker(self)
        self.tracker.run()
    
    def mapping(self, rank):
        self.mapper = Mapper(self)
        self.mapper.run()

    def run(self):
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, )) 
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        exit_codes = [p.exitcode for p in processes]
        if all(code == 0 for code in exit_codes) and int(self.end_of_dataset[0].item()) == 1:
            self.write_metrics()
        else:
            print(f"[SLAM] Run did not finish cleanly; metrics not written. exit_codes={exit_codes}")
    
    def write_metrics(self):
        metrics = {
            "tracking_avg_fps": float(self.tracking_avg_fps_shared[0].item()),
            "mapping_avg_fps": float(self.mapping_avg_fps_shared[0].item()),
            "tracking_ate_rmse": float(self.tracking_ate_rmse_shared[0].item()),
            "mapping_ate_rmse": float(self.mapping_ate_rmse_shared[0].item()),
            "final_gaussian_count": int(self.final_gaussian_count_shared[0].item()),
            "gpu_avg_util": float(self.gpu_avg_util_shared[0].item())
        }
        metrics_path = os.path.join(self.output_path, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SLAM] Saved run metrics to {metrics_path}")

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
                    net_image = render(custom_cam, self.gaussians, 0.0)["surf_depth"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                  
                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

if __name__ == "__main__":
    parser = ArgumentParser(description="GS-ICP-SLAM")
    parser.add_argument("--config_file", help="Path to configuration file", default=None)
    
    # Optional overrides
    parser.add_argument("--dataset_path", help="dataset path", default=None)
    parser.add_argument("--keyframe_th", default=None)
    parser.add_argument("--knn_maxd", default=None)
    parser.add_argument("--verbose", action='store_true', default=None)
    parser.add_argument("--demo", action='store_true', default=None)
    parser.add_argument("--overlapped_th", default=None)
    parser.add_argument("--max_correspondence_distance", default=None)
    parser.add_argument("--trackable_opacity_th", default=None)
    parser.add_argument("--overlapped_th2", default=None)
    parser.add_argument("--downsample_rate", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--keyframe_freq", default=None)
    parser.add_argument("--n_trackable_keyframes", default=None)
    parser.add_argument("--use_dynamic_fov", default=None)
    parser.add_argument("--loop_constraint_noise", default=None)
    parser.add_argument("--downsample_voxel_size", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--loop_overlap_th", default=None)

    # densify related parameters
    parser.add_argument("--use_densify", default=None)
    parser.add_argument("--densify_start_iteration", default=None)
    parser.add_argument("--densify_threshold_opacity", default=None)
    parser.add_argument("--densify_percentage", default=None)
    parser.add_argument("--opt_scaling_max", default=None)

    # pruning related parameters
    parser.add_argument("--pruning_min_opacity", default=None)
    parser.add_argument("--pruning_min_scale", default=None)

    
    # Custom optimization parameters
    parser.add_argument("--opt_lambda_dist", default=None)
    parser.add_argument("--decay_speed", default=None)
    parser.add_argument("--loop_cooldown_time", default=None)
    parser.add_argument("--rerun_viewer", default=False)
    parser.add_argument("--densify_frequency", default=None)
    
    args = parser.parse_args()

    # 1. Load Config
    cfg = load_configuration(Path(args.config_file) if args.config_file else None)
    
    # 2. Override with CLI args if provided
    if args.dataset_path: cfg.data.dataset_path = args.dataset_path
    if args.keyframe_th: cfg.slam.keyframe_th = float(args.keyframe_th)
    if args.knn_maxd: cfg.slam.knn_maxd = float(args.knn_maxd)
    if args.verbose is not None: cfg.verbose = args.verbose
    if args.demo is not None: cfg.demo = args.demo
    if args.overlapped_th: cfg.slam.overlapped_th = float(args.overlapped_th)
    if args.max_correspondence_distance: cfg.slam.max_correspondence_distance = float(args.max_correspondence_distance)
    if args.trackable_opacity_th: cfg.slam.trackable_opacity_th = float(args.trackable_opacity_th)
    if args.overlapped_th2: cfg.slam.overlapped_th2 = float(args.overlapped_th2)
    if args.downsample_rate: cfg.slam.downsample_rate = int(args.downsample_rate)
    if args.test: cfg.test = args.test
    if args.keyframe_freq: cfg.slam.keyframe_freq = int(args.keyframe_freq)
    if args.n_trackable_keyframes: cfg.slam.n_trackable_keyframes = int(args.n_trackable_keyframes)
    if args.loop_constraint_noise: cfg.slam.loop_constraint_noise = float(args.loop_constraint_noise)
    if args.downsample_voxel_size: cfg.slam.downsample_voxel_size= float(args.downsample_voxel_size)
    if args.loop_overlap_th: cfg.slam.loop_overlap_th = float(args.loop_overlap_th)
    # Override optimization parameters if provided
    if args.opt_lambda_dist: cfg.slam.opt_lambda_dist = float(args.opt_lambda_dist)
    if args.decay_speed: cfg.slam.decay_speed = float(args.decay_speed)
    if args.loop_cooldown_time: cfg.slam.loop_cooldown_time = float(args.loop_cooldown_time)
    if args.use_densify: cfg.slam.use_densify = args.use_densify.lower() in ('true', '1', 'yes', 't')

    gs_icp_slam = GS_ICP_SLAM(args, cfg)
    gs_icp_slam.run()
