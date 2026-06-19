import torch
import numpy as np


def generate_scene():
    n_points = 32
    length = 10.0
    print(f"Generating flat scene with:"
          f"{n_points}x{n_points} gaussians "
          f"aranged in a {length}x{length} [m] plane.")
    x = np.linspace(-1, 1, n_points) * length
    y = np.linspace(-1, 1, n_points) * length
    x, y = np.meshgrid(x, y)
    means3D = (
        torch.from_numpy(
            np.stack([x, y, 0 * np.ones_like(x)], axis=-1).reshape(-1, 3))
        .cuda()
        .float()
    )
    quats = torch.zeros(1, 4).repeat(len(means3D), 1).float().cuda()
    quats[..., 0] = 1.0
    scales = (length) / (n_points - 1)
    scales = torch.zeros(1, 2).repeat(
        len(means3D), 1).fill_(scales).float().cuda()
    opacity = torch.ones((means3D.shape[0], 1)).float().cuda()
    # Camera parameters
    fov_top = np.pi / 4
    fov_bottom = -np.pi / 4
    fov_right = np.pi
    fov_left = -np.pi
    hfov = fov_right - fov_left
    vfov = fov_top - fov_bottom

    vfov = np.pi / 2
    width = 2048
    height = 512
    intrins = (
        torch.Tensor(
            [
                [-width / hfov, 0, width * 0.5],
                [0, -height / vfov, height * 0.5],
                [0, 0, 1],
            ]
        )
        .float()
        .cuda()
    )
    world_T_lidar = (
        torch.Tensor([[0, -1, 0, 0], [0, 0, 1, 0.49],
                     [-1, 0, 0, 5], [0, 0, 0, 1]])
        .float()
        .cuda()
    )
    projmat = torch.zeros((4, 4)).float().cuda()
    projmat[:3, :3] = intrins
    projmat = projmat.T
    viewmat = torch.linalg.inv(world_T_lidar).T
    return (
        means3D,
        quats,
        scales,
        opacity,
        (width, height),
        (hfov, vfov),
        intrins,
        projmat,
        viewmat,
    )
