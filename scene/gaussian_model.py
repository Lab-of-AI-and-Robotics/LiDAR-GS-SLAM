#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation
import open3d as o3d
import time


class GaussianModel(nn.Module):

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def setup_functions(self):
        
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        super().__init__()
        
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        self.keyframe_idx = torch.empty(0)
        self.trackable_mask = torch.empty(0)
        self.control_score = torch.empty(0)
        self.phys_conf = torch.empty(0)
        
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def create_from_pcd2_tensor(self, points, rots_, scales_, z_vals_, trackable_idxs, control_score_=None, phys_conf_=None, keyframe_idx=0):

        fused_point_cloud = points
        z_vals = torch.clamp_min(z_vals_, 1.).unsqueeze(-1).repeat(1,2)
        scales_withz = scales_[:,:2] * z_vals * 0.1
        scales_withz = torch.clamp_max(scales_withz, 0.5)
        scales = torch.log(scales_withz)

        rots = rots_

        opacity_init_time = time.perf_counter()
        if phys_conf_ is None:
            phys_conf = torch.ones((fused_point_cloud.shape[0],), dtype=torch.float32, device="cuda")
        else:
            phys_conf = torch.clamp(phys_conf_.reshape(-1).float().cuda(), 0.0, 1.0)
            if phys_conf.shape[0] != fused_point_cloud.shape[0]:
                fixed = torch.ones((fused_point_cloud.shape[0],), dtype=torch.float32, device="cuda")
                n = min(fused_point_cloud.shape[0], phys_conf.shape[0])
                if n > 0:
                    fixed[:n] = phys_conf[:n]
                phys_conf = fixed
        alpha_min, alpha_max = 0.1, 0.9
        alpha_init = alpha_min + (alpha_max - alpha_min) * phys_conf
        opacities = inverse_sigmoid(alpha_init.unsqueeze(1))

        opacity_duration = time.perf_counter() - opacity_init_time
        print(f"Opacity initialization took {opacity_duration:.4f} seconds.")
    
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        n = fused_point_cloud.shape[0]
        empty_feat = torch.empty((n, 0, 3), device="cuda", dtype=torch.float)
        self._features_dc = nn.Parameter(empty_feat.clone().requires_grad_(True))
        self._features_rest = nn.Parameter(empty_feat.clone().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self.trackable_mask = torch.zeros((self.get_xyz.shape[0]), dtype=torch.bool, device="cuda")
        self.trackable_mask[(trackable_idxs)] = 1
        if control_score_ is None:
            self.control_score = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float32, device="cuda")
        else:
            cs = control_score_.reshape(-1).float().cuda()
            n = min(self.get_xyz.shape[0], cs.shape[0])
            self.control_score = torch.zeros((self.get_xyz.shape[0],), dtype=torch.float32, device="cuda")
            if n > 0:
                self.control_score[:n] = cs[:n]
        self.phys_conf = phys_conf

        self.keyframe_idx = torch.ones((self.get_xyz.shape[0],1), dtype=torch.int32, device="cuda")*keyframe_idx
        
        torch.cuda.empty_cache()
    
    def add_from_pcd2_tensor(self, points, rots_, scales_, z_vals_, trackable_idxs, control_score_=None, phys_conf_=None, keyframe_idx=0):

        fused_point_cloud = points

        z_vals = torch.clamp_min(z_vals_, 1.).unsqueeze(-1).repeat(1,2)
        scales_withz = scales_[:,:2] * z_vals * 0.1
        scales_withz = torch.clamp_max(scales_withz, 0.5)
        scales = torch.log(scales_withz)

        rots = rots_
        if phys_conf_ is None:
            self.new_phys_conf = torch.ones((fused_point_cloud.shape[0],), dtype=torch.float32, device="cuda")
        else:
            phys = torch.clamp(phys_conf_.reshape(-1).float().cuda(), 0.0, 1.0)
            if phys.shape[0] != fused_point_cloud.shape[0]:
                fixed = torch.ones((fused_point_cloud.shape[0],), dtype=torch.float32, device="cuda")
                n = min(fused_point_cloud.shape[0], phys.shape[0])
                if n > 0:
                    fixed[:n] = phys[:n]
                phys = fixed
            self.new_phys_conf = phys
        alpha_min, alpha_max = 0.1, 0.9
        alpha_init = alpha_min + (alpha_max - alpha_min) * self.new_phys_conf
        opacities = inverse_sigmoid(alpha_init.unsqueeze(1))

        self.new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        n = fused_point_cloud.shape[0]
        empty_feat = torch.empty((n, 0, 3), device="cuda", dtype=torch.float)
        self.new_features_dc = nn.Parameter(empty_feat.clone().requires_grad_(True))
        self.new_features_rest = nn.Parameter(empty_feat.clone().requires_grad_(True))
        self.new_scaling = nn.Parameter(scales.requires_grad_(True))
        self.new_rotation = nn.Parameter(rots.requires_grad_(True))
        self.new_opacities = nn.Parameter(opacities.requires_grad_(True))
        
        
        self.new_trackable_mask = torch.zeros((self.new_xyz.shape[0]), dtype=torch.bool, device="cuda")
        if len(trackable_idxs) != 0:
            self.new_trackable_mask[(trackable_idxs)] = 1
        if control_score_ is None:
            self.new_control_score = torch.zeros((self.new_xyz.shape[0],), dtype=torch.float32, device="cuda")
        else:
            cs = control_score_.reshape(-1).float().cuda()
            n = min(self.new_xyz.shape[0], cs.shape[0])
            self.new_control_score = torch.zeros((self.new_xyz.shape[0],), dtype=torch.float32, device="cuda")
            if n > 0:
                self.new_control_score[:n] = cs[:n]

        self.densification_postfix(self.new_xyz, self.new_features_dc, 
                                   self.new_features_rest, self.new_opacities,
                                   self.new_scaling, self.new_rotation, self.new_trackable_mask,
                                   self.new_control_score, self.new_phys_conf)

        new_keyframe_idx = torch.ones((self.new_xyz.shape[0], self.keyframe_idx.shape[1]), device="cuda", dtype=torch.int32)*keyframe_idx
        self.keyframe_idx = torch.concat([  self.keyframe_idx,
                                            new_keyframe_idx], dim=0)
        
        torch.cuda.empty_cache()
        
       
    def get_trackable_gaussians_tensor_trackble(self, opacity_th, trackable_kf_idx, conf_th=0.0):

        with torch.no_grad():
            opacity_filter = self.get_opacity > opacity_th
           
            idx_filter = self.keyframe_idx > trackable_kf_idx
            n = int(self.get_xyz.shape[0])
            phys_conf = self._phys_conf_vector(n)
            track_conf = self.get_opacity.squeeze(-1) * phys_conf
            conf_filter = track_conf > conf_th
            idx_mask = idx_filter.squeeze(-1)
            opacity_mask = opacity_filter.squeeze(-1)
            base_mask = torch.logical_and(self.trackable_mask, idx_mask)
            base_opacity_mask = torch.logical_and(base_mask, opacity_mask)
            target_idxs = torch.logical_and(base_opacity_mask, conf_filter)

            target_points = self.get_xyz[target_idxs]
            target_rots = self.get_rotation[target_idxs]
            target_scales = self.get_scaling[target_idxs]
            target_control_score = self.control_score[target_idxs]
            
            
            # Pad 2D scales to 3D for GICP tracker compatibility
            if target_scales.shape[1] == 2:
                # Add a small third dimension to maintain surfel shape
                third_dim = torch.ones((target_scales.shape[0], 1), device=target_scales.device) * 1e-15
                target_scales = torch.cat([target_scales, third_dim], dim=1)
            
            return target_points.cpu(), target_rots.cpu(), target_scales.cpu(), target_control_score.cpu()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_pcd_for_mesh(self, path, opacity_threshold=0.8):
        with torch.no_grad():
            xyz = self.get_xyz
            opacity = self.get_opacity
            scaling = self.get_scaling
            rotation = self.get_rotation
            
            mask = (opacity > opacity_threshold).squeeze()
            
            if mask.sum() == 0:
                print("[Warning] No points passed the opacity threshold.")
                return

            filtered_xyz = xyz[mask].cpu().numpy()
            filtered_scaling = scaling[mask].cpu().numpy()
            filtered_rotation = rotation[mask].cpu().numpy()
            
            from scipy.spatial.transform import Rotation as R
            
            r = R.from_quat(filtered_rotation, scalar_first=True) 
            rot_matrices = r.as_matrix() 
            
            min_scale_idx = np.argmin(filtered_scaling, axis=1)
            
            normals = np.zeros_like(filtered_xyz)
            for i in range(len(filtered_xyz)):
                normals[i] = rot_matrices[i, :, min_scale_idx[i]]
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
            
            o3d.io.write_point_cloud(path, pcd)
            print(f"[GaussianModel] Mesh-ready PCD saved to {path} ({len(filtered_xyz)} points)")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.trackable_mask = self.trackable_mask[valid_points_mask]
        if self.control_score.numel() > 0:
            self.control_score = self.control_score[valid_points_mask]
        if self.phys_conf.numel() > 0:
            self.phys_conf = self.phys_conf[valid_points_mask]

        
        try:
            self.keyframe_idx = self.keyframe_idx[valid_points_mask]
        except:
            pass

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_trackable_mask=None, new_control_score=None, new_phys_conf=None):

        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if new_trackable_mask is None:
            new_trackable_mask = torch.zeros((new_xyz.shape[0],), dtype=torch.bool, device="cuda")
        self.trackable_mask = torch.concat([self.trackable_mask, new_trackable_mask], dim=0)
        if new_control_score is None:
            new_control_score = torch.zeros((new_xyz.shape[0],), dtype=torch.float32, device="cuda")
        else:
            new_control_score = new_control_score.reshape(-1).float().cuda()
            if new_control_score.shape[0] != new_xyz.shape[0]:
                fixed = torch.zeros((new_xyz.shape[0],), dtype=torch.float32, device="cuda")
                n = min(new_xyz.shape[0], new_control_score.shape[0])
                if n > 0:
                    fixed[:n] = new_control_score[:n]
                new_control_score = fixed
        if self.control_score.numel() == 0:
            self.control_score = new_control_score
        else:
            self.control_score = torch.concat([self.control_score, new_control_score], dim=0)
        if new_phys_conf is None:
            new_phys_conf = torch.ones((new_xyz.shape[0],), dtype=torch.float32, device="cuda")
        else:
            new_phys_conf = torch.clamp(new_phys_conf.reshape(-1).float().cuda(), 0.0, 1.0)
            if new_phys_conf.shape[0] != new_xyz.shape[0]:
                fixed = torch.ones((new_xyz.shape[0],), dtype=torch.float32, device="cuda")
                n = min(new_xyz.shape[0], new_phys_conf.shape[0])
                if n > 0:
                    fixed[:n] = new_phys_conf[:n]
                new_phys_conf = fixed
        if self.phys_conf.numel() == 0:
            self.phys_conf = new_phys_conf
        else:
            self.phys_conf = torch.concat([self.phys_conf, new_phys_conf], dim=0)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        if scene_extent != None:
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        samples = torch.zeros((stds.size(0), 3), device="cuda")
        samples[:, :2] = torch.normal(
            mean=torch.zeros_like(stds),
            std=stds
            )
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_trackable_mask = self.trackable_mask[selected_pts_mask].repeat(N)
        new_keyframe_idx = None
        if hasattr(self, "keyframe_idx") and isinstance(self.keyframe_idx, torch.Tensor) and self.keyframe_idx.numel() > 0:
            new_keyframe_idx = self.keyframe_idx[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_trackable_mask)
        if new_keyframe_idx is not None:
            self.keyframe_idx = torch.concat([self.keyframe_idx, new_keyframe_idx], dim=0)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def _control_score_vector(self, n_ref=None):
        n = self.get_xyz.shape[0] if n_ref is None else int(n_ref)
        if self.control_score.numel() == n:
            return self.control_score.reshape(-1).float()

        cs = torch.zeros((n,), dtype=torch.float32, device="cuda")
        if self.control_score.numel() > 0:
            src = self.control_score.reshape(-1).float().cuda()
            m = min(n, src.shape[0])
            if m > 0:
                cs[:m] = src[:m]
        return cs

    def _phys_conf_vector(self, n_ref=None):
        n = self.get_xyz.shape[0] if n_ref is None else int(n_ref)
        if self.phys_conf.numel() == n:
            return self.phys_conf.reshape(-1).float()

        pc = torch.ones((n,), dtype=torch.float32, device="cuda")
        if self.phys_conf.numel() > 0:
            src = self.phys_conf.reshape(-1).float().cuda()
            m = min(n, src.shape[0])
            if m > 0:
                pc[:m] = src[:m]
        return pc


    def _cap_mask_by_score(self, mask, score, max_k):
        if max_k <= 0:
            return torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
        if int(mask.sum().item()) <= max_k:
            return mask
        masked_score = score.clone()
        masked_score[~mask] = -1e9
        topk_idx = torch.topk(masked_score, k=max_k).indices
        out = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
        out[topk_idx] = True
        return out

    def _select_control_masks(self, scene_extent, split_q=0.95, clone_q=0.75, max_split_ratio=0.0005, max_clone_ratio=0.05):
        n0 = self.get_xyz.shape[0]
        if n0 <= 0:
            z = torch.zeros((0,), dtype=torch.bool, device="cuda")
            return z, z, {"t_split": 1.0, "t_clone": 1.0, "num_split": 0, "num_clone": 0}

        cs = self._control_score_vector(n0)
        valid = torch.isfinite(cs)
        if int(valid.sum().item()) < 32:
            z = torch.zeros((n0,), dtype=torch.bool, device="cuda")
            return z, z, {"t_split": 1.0, "t_clone": 1.0, "num_split": 0, "num_clone": 0}

        vals = cs[valid]
        t_split = float(torch.quantile(vals, split_q).item())
        t_clone = float(torch.quantile(vals, clone_q).item())
        if t_clone > t_split:
            t_clone = t_split

        split_mask = valid & (cs >= t_split)
        clone_mask = valid & (cs >= t_clone) & (cs < t_split)

        if scene_extent is not None:
            scale_max = torch.max(self.get_scaling, dim=1).values
            small_mask = scale_max <= self.percent_dense * scene_extent
            large_mask = ~small_mask
            clone_mask = clone_mask & small_mask
            split_mask = split_mask & large_mask

        max_split = max(1, int(n0 * max_split_ratio))
        max_clone = max(1, int(n0 * max_clone_ratio))
        split_mask = self._cap_mask_by_score(split_mask, cs, max_split)
        clone_mask = self._cap_mask_by_score(clone_mask, cs, max_clone)

        # Ensure disjoint masks
        clone_mask = clone_mask & (~split_mask)

        return split_mask, clone_mask, {
            "t_split": t_split,
            "t_clone": t_clone,
            "num_split": int(split_mask.sum().item()),
            "num_clone": int(clone_mask.sum().item()),
        }

    def _plane_cover_and_prune_mask(
        self,
        min_opacity,
        voxel_size,
        plane_cs_q,
        min_group_size,
        cover_q,
        cover_margin,
        max_expand_ratio,
        op_prune_cap,
        max_candidates,
        max_prune_ratio,
        group_axes=(0, 1),
    ):
        with torch.no_grad():
            n = int(self.get_xyz.shape[0])
            plane_prune_mask = torch.zeros((n,), dtype=torch.bool, device="cuda")
            if not isinstance(group_axes, (tuple, list)) or len(group_axes) != 2:
                group_axes = (0, 1)
            ax0, ax1 = int(group_axes[0]), int(group_axes[1])
            if ax0 < 0 or ax0 > 2 or ax1 < 0 or ax1 > 2 or ax0 == ax1:
                ax0, ax1 = 0, 1
            stats = {
                "t_plane": 0.0,
                "num_candidates": 0,
                "num_groups": 0,
                "num_expanded": 0,
                "num_plane_pruned": 0,
                "group_axes": (ax0, ax1),
                "expanded_old_major_mean": 0.0,
                "expanded_old_major_p50": 0.0,
                "expanded_old_major_p90": 0.0,
                "expanded_new_major_mean": 0.0,
                "expanded_new_major_p50": 0.0,
                "expanded_new_major_p90": 0.0,
                "expanded_major_ratio_mean": 0.0,
                "expanded_major_ratio_p50": 0.0,
                "expanded_major_ratio_p90": 0.0,
                "expanded_old_area_mean": 0.0,
                "expanded_new_area_mean": 0.0,
                "expanded_area_ratio_mean": 0.0,
            }

            if n == 0:
                return plane_prune_mask, stats

            if voxel_size <= 0:
                return plane_prune_mask, stats

            cs = self._control_score_vector(n)
            finite = torch.isfinite(cs)
            positive = finite & (cs > 0.0)
            if int(positive.sum().item()) < 16:
                return plane_prune_mask, stats

            t_plane = float(torch.quantile(cs[positive], plane_cs_q).item())
            stats["t_plane"] = t_plane

            opacity = self.get_opacity.squeeze(-1)
            cand_mask = positive & (cs <= t_plane) & (opacity >= min_opacity)
            cand_idx = torch.where(cand_mask)[0]
            if cand_idx.numel() == 0:
                return plane_prune_mask, stats

            if cand_idx.numel() > max_candidates:
                order = torch.argsort(cs[cand_idx], descending=False)
                cand_idx = cand_idx[order[:max_candidates]]
            stats["num_candidates"] = int(cand_idx.numel())

            xyz = self.get_xyz.detach()
            rots_q = self._rotation.detach()
            scales_lin = self.get_scaling.detach().clone()

            cand_xyz = xyz[cand_idx]
            # 2-axis voxel grouping (e.g., XY or XZ)
            vox_xy = torch.floor(cand_xyz[:, [ax0, ax1]] / voxel_size).to(torch.int64).cpu().numpy()
            groups = {}
            for local_i, v in enumerate(vox_xy):
                key = (int(v[0]), int(v[1]))
                if key not in groups:
                    groups[key] = []
                groups[key].append(local_i)

            max_prune = max(1, int(max_prune_ratio * n))
            num_groups = 0
            num_expanded = 0
            num_plane_pruned = 0
            expanded_old_major = []
            expanded_new_major = []
            expanded_major_ratio = []
            expanded_old_area = []
            expanded_new_area = []
            expanded_area_ratio = []

            for local_ids in groups.values():
                if len(local_ids) < min_group_size:
                    continue
                num_groups += 1

                local_tensor = torch.as_tensor(local_ids, device="cuda", dtype=torch.long)
                gidx = cand_idx[local_tensor]
                g_xyz = xyz[gidx]
                g_opacity = opacity[gidx]
                g_cs = cs[gidx]

                rep = gidx[torch.argmax(g_opacity)]
                rep_xyz = xyz[rep]

                R = build_rotation(rots_q[rep:rep+1])[0]
                u = R[:, 0]
                v = R[:, 1]

                delta = g_xyz - rep_xyz.unsqueeze(0)
                du = torch.abs(delta @ u)
                dv = torch.abs(delta @ v)

                su = torch.quantile(du, cover_q).clamp_min(1e-4)
                sv = torch.quantile(dv, cover_q).clamp_min(1e-4)

                dim_uv = min(2, int(scales_lin.shape[1]))
                if dim_uv > 0:
                    old_uv = scales_lin[rep, :dim_uv]
                    target_uv = torch.tensor([su, sv], device="cuda", dtype=old_uv.dtype)[:dim_uv]
                    new_uv = torch.minimum(torch.maximum(old_uv, target_uv), old_uv * max_expand_ratio)
                    if torch.any(new_uv > old_uv * 1.001):
                        old_major = float(torch.max(old_uv).item())
                        new_major = float(torch.max(new_uv).item())
                        old_area = float(torch.prod(old_uv).item()) if dim_uv >= 2 else old_major
                        new_area = float(torch.prod(new_uv).item()) if dim_uv >= 2 else new_major

                        old_log = self._scaling.data[rep, :dim_uv]
                        self._scaling.data[rep, :dim_uv] = torch.log(new_uv.clamp_min(1e-6)).to(old_log.dtype)
                        num_expanded += 1
                        expanded_old_major.append(old_major)
                        expanded_new_major.append(new_major)
                        expanded_major_ratio.append(new_major / max(old_major, 1e-12))
                        expanded_old_area.append(old_area)
                        expanded_new_area.append(new_area)
                        expanded_area_ratio.append(new_area / max(old_area, 1e-12))

                cover = (du <= cover_margin * su) & (dv <= cover_margin * sv)
                prune_local = cover & (g_cs <= t_plane)
                prune_idx = gidx[prune_local]
                prune_idx = prune_idx[prune_idx != rep]
                if prune_idx.numel() > 0:
                    remain = max_prune - int(plane_prune_mask.sum().item())
                    if remain <= 0:
                        break
                    if prune_idx.numel() > remain:
                        prune_idx = prune_idx[:remain]
                    plane_prune_mask[prune_idx] = True
                    num_plane_pruned += int(prune_idx.numel())

            stats["num_groups"] = num_groups
            stats["num_expanded"] = num_expanded
            stats["num_plane_pruned"] = num_plane_pruned
            if len(expanded_old_major) > 0:
                old_major_arr = np.asarray(expanded_old_major, dtype=np.float64)
                new_major_arr = np.asarray(expanded_new_major, dtype=np.float64)
                major_ratio_arr = np.asarray(expanded_major_ratio, dtype=np.float64)
                old_area_arr = np.asarray(expanded_old_area, dtype=np.float64)
                new_area_arr = np.asarray(expanded_new_area, dtype=np.float64)
                area_ratio_arr = np.asarray(expanded_area_ratio, dtype=np.float64)

                stats["expanded_old_major_mean"] = float(np.mean(old_major_arr))
                stats["expanded_old_major_p50"] = float(np.percentile(old_major_arr, 50))
                stats["expanded_old_major_p90"] = float(np.percentile(old_major_arr, 90))
                stats["expanded_new_major_mean"] = float(np.mean(new_major_arr))
                stats["expanded_new_major_p50"] = float(np.percentile(new_major_arr, 50))
                stats["expanded_new_major_p90"] = float(np.percentile(new_major_arr, 90))
                stats["expanded_major_ratio_mean"] = float(np.mean(major_ratio_arr))
                stats["expanded_major_ratio_p50"] = float(np.percentile(major_ratio_arr, 50))
                stats["expanded_major_ratio_p90"] = float(np.percentile(major_ratio_arr, 90))
                stats["expanded_old_area_mean"] = float(np.mean(old_area_arr))
                stats["expanded_new_area_mean"] = float(np.mean(new_area_arr))
                stats["expanded_area_ratio_mean"] = float(np.mean(area_ratio_arr))
            return plane_prune_mask, stats

    def plane_surface_manage(
        self,
        min_opacity,
        voxel_size=1.0,
        plane_cs_q=0.25,
        min_group_size=3,
        cover_q=0.90,
        cover_margin=1.1,
        max_expand_ratio=1.1,
        op_prune_cap=None,
        max_candidates=20000,
        max_prune_ratio=0.01,
        enable_xz_pass=True,
        xz_pass_ratio=0.3,
        xz_voxel_size=None,
        xz_max_candidates=None,
        enable_yz_pass=True,
        yz_pass_ratio=0.2,
        yz_voxel_size=None,
        yz_max_candidates=None,
    ):
        with torch.no_grad():
            if op_prune_cap is None:
                op_prune_cap = float(min_opacity) + 0.03

            num_total_before = int(self.get_xyz.shape[0])
            plane_prune_mask_xy, stats_xy = self._plane_cover_and_prune_mask(
                min_opacity=min_opacity,
                voxel_size=voxel_size,
                plane_cs_q=plane_cs_q,
                min_group_size=min_group_size,
                cover_q=cover_q,
                cover_margin=cover_margin,
                max_expand_ratio=max_expand_ratio,
                op_prune_cap=op_prune_cap,
                max_candidates=max_candidates,
                max_prune_ratio=max_prune_ratio,
                group_axes=(0, 1),
            )

            num_plane_pruned_xy = int(plane_prune_mask_xy.sum().item())
            if num_plane_pruned_xy > 0:
                self.prune_points(plane_prune_mask_xy)

            stats_xz = {
                "t_plane": 0.0,
                "num_candidates": 0,
                "num_groups": 0,
                "num_expanded": 0,
                "num_plane_pruned": 0,
            }
            num_plane_pruned_xz = 0
            xz_voxel = voxel_size if xz_voxel_size is None else float(xz_voxel_size)
            xz_candidates = max_candidates if xz_max_candidates is None else int(xz_max_candidates)
            xz_ratio = max(0.0, float(xz_pass_ratio))
            if enable_xz_pass and xz_ratio > 0.0 and int(self.get_xyz.shape[0]) > 0:
                plane_prune_mask_xz, stats_xz = self._plane_cover_and_prune_mask(
                    min_opacity=min_opacity,
                    voxel_size=xz_voxel,
                    plane_cs_q=plane_cs_q,
                    min_group_size=min_group_size,
                    cover_q=cover_q,
                    cover_margin=cover_margin,
                    max_expand_ratio=max_expand_ratio,
                    op_prune_cap=op_prune_cap,
                    max_candidates=xz_candidates,
                    max_prune_ratio=max_prune_ratio * xz_ratio,
                    group_axes=(0, 2),
                )
                num_plane_pruned_xz = int(plane_prune_mask_xz.sum().item())
                if num_plane_pruned_xz > 0:
                    self.prune_points(plane_prune_mask_xz)

            stats_yz = {
                "t_plane": 0.0,
                "num_candidates": 0,
                "num_groups": 0,
                "num_expanded": 0,
                "num_plane_pruned": 0,
            }
            num_plane_pruned_yz = 0
            yz_voxel = voxel_size if yz_voxel_size is None else float(yz_voxel_size)
            yz_candidates = max_candidates if yz_max_candidates is None else int(yz_max_candidates)
            yz_ratio = max(0.0, float(yz_pass_ratio))
            if enable_yz_pass and yz_ratio > 0.0 and int(self.get_xyz.shape[0]) > 0:
                plane_prune_mask_yz, stats_yz = self._plane_cover_and_prune_mask(
                    min_opacity=min_opacity,
                    voxel_size=yz_voxel,
                    plane_cs_q=plane_cs_q,
                    min_group_size=min_group_size,
                    cover_q=cover_q,
                    cover_margin=cover_margin,
                    max_expand_ratio=max_expand_ratio,
                    op_prune_cap=op_prune_cap,
                    max_candidates=yz_candidates,
                    max_prune_ratio=max_prune_ratio * yz_ratio,
                    group_axes=(1, 2),
                )
                num_plane_pruned_yz = int(plane_prune_mask_yz.sum().item())
                if num_plane_pruned_yz > 0:
                    self.prune_points(plane_prune_mask_yz)

            num_plane_pruned = num_plane_pruned_xy + num_plane_pruned_xz + num_plane_pruned_yz
            stats = dict(stats_xy)
            stats["num_plane_pruned"] = num_plane_pruned
            stats["num_total_before"] = num_total_before
            stats["num_total_after"] = int(self.get_xyz.shape[0])
            stats["num_plane_pruned_applied"] = num_plane_pruned
            stats["num_plane_pruned_xy"] = num_plane_pruned_xy
            stats["num_plane_pruned_xz"] = num_plane_pruned_xz
            stats["num_plane_pruned_yz"] = num_plane_pruned_yz
            stats["num_candidates_xy"] = int(stats_xy.get("num_candidates", 0))
            stats["num_candidates_xz"] = int(stats_xz.get("num_candidates", 0))
            stats["num_candidates_yz"] = int(stats_yz.get("num_candidates", 0))
            stats["num_groups_xy"] = int(stats_xy.get("num_groups", 0))
            stats["num_groups_xz"] = int(stats_xz.get("num_groups", 0))
            stats["num_groups_yz"] = int(stats_yz.get("num_groups", 0))
            stats["num_expanded_xy"] = int(stats_xy.get("num_expanded", 0))
            stats["num_expanded_xz"] = int(stats_xz.get("num_expanded", 0))
            stats["num_expanded_yz"] = int(stats_yz.get("num_expanded", 0))
            stats["num_expanded"] = int(
                stats.get("num_expanded_xy", 0)
                + stats.get("num_expanded_xz", 0)
                + stats.get("num_expanded_yz", 0)
            )
            stats["xz_pass_enabled"] = bool(enable_xz_pass and xz_ratio > 0.0)
            stats["yz_pass_enabled"] = bool(enable_yz_pass and yz_ratio > 0.0)
            return stats

    def densify_and_split_control(self, selected_pts_mask, N=2):
        n = self.get_xyz.shape[0]
        if selected_pts_mask.shape[0] != n:
            aligned = torch.zeros((n,), dtype=torch.bool, device="cuda")
            m = min(n, selected_pts_mask.shape[0])
            if m > 0:
                aligned[:m] = selected_pts_mask[:m]
            selected_pts_mask = aligned

        selected_count = int(selected_pts_mask.sum().item())
        if selected_count == 0:
            return 0

        parent_scaling = self.get_scaling[selected_pts_mask]  # [M, D], D is typically 2
        parent_xyz = self.get_xyz[selected_pts_mask]          # [M, 3]
        parent_rots = self._rotation[selected_pts_mask]       # [M, 4]
        M = parent_scaling.shape[0]
        D = min(parent_scaling.shape[1], 3)

        # Deterministic offsets: split along each parent's major local axis.
        major_idx = torch.argmax(parent_scaling[:, :D], dim=1)  # [M]
        major_scale = parent_scaling[:, :D].gather(1, major_idx.unsqueeze(1)).squeeze(1)  # [M]
        d = 0.008 * major_scale

        if N == 2:
            coeff = torch.tensor([-1.0, 1.0], dtype=parent_scaling.dtype, device="cuda")
        else:
            coeff = torch.linspace(-1.0, 1.0, steps=N, dtype=parent_scaling.dtype, device="cuda")

        offset_mag = d[:, None] * coeff[None, :]  # [M, N]
        local_offsets = torch.zeros((M, N, 3), dtype=parent_scaling.dtype, device="cuda")
        for axis in range(D):
            axis_mask = (major_idx == axis)
            if bool(axis_mask.any()):
                local_offsets[axis_mask, :, axis] = offset_mag[axis_mask, :]

        samples = local_offsets.reshape(-1, 3)  # [M*N, 3]
        rots = build_rotation(parent_rots).repeat_interleave(N, dim=0)  # [M*N, 3, 3]
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + parent_xyz.repeat_interleave(N, dim=0)

        new_scaling = self.scaling_inverse_activation(parent_scaling.repeat_interleave(N, dim=0) / (0.8 * N))
        new_rotation = parent_rots.repeat_interleave(N, dim=0)
        new_features_dc = self._features_dc[selected_pts_mask].repeat_interleave(N, dim=0)
        new_features_rest = self._features_rest[selected_pts_mask].repeat_interleave(N, dim=0)
        new_opacity = self._opacity[selected_pts_mask].repeat_interleave(N, dim=0)
        new_trackable_mask = self.trackable_mask[selected_pts_mask].repeat_interleave(N, dim=0)
        new_control_score = self._control_score_vector(n)[selected_pts_mask].repeat_interleave(N, dim=0)
        new_phys_conf = self._phys_conf_vector(n)[selected_pts_mask].repeat_interleave(N, dim=0)

        new_keyframe_idx = None
        if hasattr(self, "keyframe_idx") and isinstance(self.keyframe_idx, torch.Tensor) and self.keyframe_idx.numel() > 0:
            new_keyframe_idx = self.keyframe_idx[selected_pts_mask].repeat_interleave(N, dim=0)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_trackable_mask,
            new_control_score,
            new_phys_conf,

        )
        if new_keyframe_idx is not None:
            self.keyframe_idx = torch.concat([self.keyframe_idx, new_keyframe_idx], dim=0)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_count, device="cuda", dtype=torch.bool))
        )
        self.prune_points(prune_filter)
        return selected_count

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        split_mask, clone_mask, score_stats = self._select_control_masks(extent)
        self.densify_and_split_control(split_mask, N=2)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        self.prune_points(prune_mask)

    def transform_gaussians(self, kf_idx, transform):
        select_filter = self.keyframe_idx == kf_idx
        select_filter = select_filter.squeeze(-1)
        
        all_xyz = self._xyz.clone()  # [N,3]
        selected_xyz = all_xyz[select_filter]
        
        all_quats = self.get_rotation.clone()
        selected_quats = all_quats[select_filter]

        transform = torch.tensor(transform, dtype=selected_xyz.dtype).cuda()
        transform_R = transform[:3,:3]
        transform_t = transform[:3,3]

        transformed_xyz = (transform_R @ selected_xyz.T).T + transform_t.unsqueeze(0)

        transformed_quats = self.apply_rotation_to_quaternions(selected_quats, transform_R)

        all_xyz[select_filter] = transformed_xyz
        optimizable_tensors_xyz = self.replace_tensor_to_optimizer(all_xyz, "xyz")

        all_quats[select_filter] = transformed_quats
        optimizable_tensors_quats = self.replace_tensor_to_optimizer(all_quats, "rotation")

        self._xyz = optimizable_tensors_xyz["xyz"]
        self._rotation = optimizable_tensors_quats["rotation"]

    def apply_rotation_to_quaternions(self, selected_quat, transform_R):
        """
        selected_quat: [N,4] (x,y,z,w)
        transform_R: [3,3]
        """
        # Convert rotation matrix to quaternion.
        def rot_to_quat(R):
            tr = R.trace()
            if tr > 0:
                S = torch.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2,1] - R[1,2]) / S
                qy = (R[0,2] - R[2,0]) / S
                qz = (R[1,0] - R[0,1]) / S
            else:
                if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
                    S = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                    qw = (R[2,1] - R[1,2]) / S
                    qx = 0.25 * S
                    qy = (R[0,1] + R[1,0]) / S
                    qz = (R[0,2] + R[2,0]) / S
                elif R[1,1] > R[2,2]:
                    S = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                    qw = (R[0,2] - R[2,0]) / S
                    qx = (R[0,1] + R[1,0]) / S
                    qy = 0.25 * S
                    qz = (R[1,2] + R[2,1]) / S
                else:
                    S = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                    qw = (R[1,0] - R[0,1]) / S
                    qx = (R[0,2] + R[2,0]) / S
                    qy = (R[1,2] + R[2,1]) / S
                    qz = 0.25 * S
            return torch.tensor([qx, qy, qz, qw], device=R.device, dtype=R.dtype)

        qR = rot_to_quat(transform_R)  # [4]

        # quaternion product (qR ⊗ Q), xyzw order
        # qR: [4], selected_quat: [N,4]
        x1, y1, z1, w1 = qR
        x2, y2, z2, w2 = selected_quat.T

        new_q = torch.stack([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dim=1)

        # normalize to unit quaternion
        new_q = new_q / new_q.norm(dim=1, keepdim=True)

        return new_q
