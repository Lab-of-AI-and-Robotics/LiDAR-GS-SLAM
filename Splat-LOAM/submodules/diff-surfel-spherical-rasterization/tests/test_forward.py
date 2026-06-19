import torch
from generate_input import generate_scene
import matplotlib.pyplot as plt
from diff_surfel_spherical_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer
)


def test_raster():
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
        debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = (
        torch.zeros_like(means3D).float().cuda() + 0
    )

    radii, allmap = rasterizer.forward(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        scales=scales,
        rotations=quats,
        cov3D_precomp=None,
    )

    fig, axs = plt.subplots(3, 1)
    rend_normal = (allmap[2:5] + 1) / 2
    rend_depth = allmap[0:1]
    rend_alpha = allmap[1:2]

    depth_exp = rend_depth / rend_alpha
    depth_exp = torch.nan_to_num(depth_exp, 0, 0)

    image_depth = depth_exp[0].cpu().numpy()
    image_alpha = rend_alpha[0].cpu().numpy()
    image_normal = rend_normal.permute(1, 2, 0).cpu().numpy()

    axs[0].imshow(image_depth, "gray")
    axs[0].set_title("Range [Depth]")
    axs[1].imshow(image_alpha)
    axs[1].set_title("Accumulated Alpha")
    axs[2].imshow(image_normal)
    axs[2].set_title("Normals")
    plt.tight_layout()
    print("Saving figure in test_forward.png")
    plt.savefig("test_forward.png", dpi=300)


if __name__ == "__main__":
    test_raster()
