import torch
import numpy as np
from diff_surfel_spherical_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)
from generate_input import generate_scene
import time


def test_backward():
    torch.autograd.set_detect_anomaly(True)
    (
        means3D,
        quats,
        scales,
        opacity,
        img_size,
        fov,
        intrins,
        projmat,
        viewmat,
    ) = generate_scene()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(img_size[1]),
        image_width=int(img_size[0]),
        scale_modifier=1.0,
        viewmatrix=viewmat,
        projmatrix=projmat,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = (
        torch.zeros_like(means3D).float().cuda() + 0
    )

    opt_params = {
        "means3D": torch.nn.Parameter(means3D, requires_grad=True),
        "means2D": torch.nn.Parameter(means2D, requires_grad=True),
        "rots": torch.nn.Parameter(quats, requires_grad=True),
        "scales": torch.nn.Parameter(scales, requires_grad=True),
        "opacity": torch.nn.Parameter(opacity, requires_grad=True)
    }
    W, H = img_size[0], img_size[1]
    opt = torch.optim.Adam(opt_params.values(), lr=0.001)
    gt_depth = torch.zeros((1, H, W)).float().cuda()
    gt_depth[:, :, W // 2 - 200: W // 2 + 200] = 1.0
    gt_alpha = gt_depth.clone()

    for i in range(3000):
        loop_start = time.time()
        opt.zero_grad()
        radii, allmap = rasterizer.forward(
            means3D=opt_params["means3D"],
            means2D=opt_params["means2D"],
            opacities=opt_params["opacity"],
            scales=opt_params["scales"],
            rotations=opt_params["rots"],
            cov3D_precomp=None,
        )
        rend_depth = allmap[0:1]
        rend_alpha = allmap[1:2]
        valid_mask = rend_alpha > 0.0
        torch.cuda.synchronize()
        loop_end_raster = time.time()

        depth_exp = rend_depth
        depth_exp[valid_mask] = depth_exp[valid_mask] / rend_alpha[valid_mask]

        loss_depth = torch.abs(depth_exp - gt_depth).mean()
        loss_alpha = torch.nn.functional.binary_cross_entropy(
            rend_alpha, gt_alpha)
        loss = loss_depth + 0.2 * loss_alpha
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        loop_end = time.time()

        print(f"loss={loss.item():.4f} | "
              f"dt={1e3 * (loop_end - loop_start):.3f} ms | "
              f"t_Rast={1000*(loop_end_raster - loop_start):.3f} ms | "
              f"t_Opt={1e3*(loop_end - loop_end_raster):.3f} ms")


if __name__ == "__main__":
    test_backward()
