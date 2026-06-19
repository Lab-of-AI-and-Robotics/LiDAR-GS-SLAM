import torch
import numpy as np

class VoxelDynamicFilter:
    def __init__(self, voxel_size=0.3, device='cuda'):
        self.voxel_size = voxel_size
        self.device = device
        self.map_voxels = set()
        self.updated = False

    def update_map(self, map_points_tensor):

        if map_points_tensor is None or len(map_points_tensor) == 0:
            return

        voxels = (map_points_tensor / self.voxel_size).floor().int()
        
        unique_voxels = torch.unique(voxels, dim=0).cpu().numpy()
        
        self.map_voxels = set(map(tuple, unique_voxels))
        self.updated = True

    def filter(self, current_points_tensor, current_pose_tensor, K):

        num_points = current_points_tensor.shape[0]
        point_ids = torch.zeros(num_points, dtype=torch.int32, device=self.device)
        
        if not self.updated or len(self.map_voxels) == 0:
            return point_ids
        curr_voxels = (current_points_tensor / self.voxel_size).floor().int().cpu().numpy()
        
        is_empty_space = [tuple(v) not in self.map_voxels for v in curr_voxels]
        is_empty_space_tensor = torch.tensor(is_empty_space, device=self.device, dtype=torch.bool)
        
        c2w = current_pose_tensor
        w2c = torch.inverse(c2w)
        
        # R: (3,3), T: (3,1)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        # P_cam = R * P_world + T
        pts_cam = (R @ current_points_tensor.T).T + T
        
        current_depths = torch.norm(pts_cam, dim=1)
        point_ids[is_empty_space_tensor] = 1
        
        return point_ids