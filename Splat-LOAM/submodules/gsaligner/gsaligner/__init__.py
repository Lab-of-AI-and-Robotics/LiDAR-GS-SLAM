import torch
from typing_extensions import Tuple
from dataclasses import dataclass
from gsaligner._C import GSAligner as GSAlignerImpl


@dataclass
class GSAlignerParams:
    image_height: int = 0
    image_width: int = 0
    geom_b_min: float = 0.1
    geom_b_max: float = 0.2
    geom_b_ratio: float = 0.02
    geom_rho_kernel: float = 0.8
    geom_iterations: int = 10
    photo_omega_depth: float = 5.0
    photo_depth_rejection_threshold: float = 0.8
    photo_rho_kernel: float = 1.0
    photo_min_depth: float = 0.5
    photo_max_depth: float = 80.0
    photo_iterations: int = 10


class GSAligner:
    """
    GSAligner is a wrapper that provides geometric and photometric pair-wise alignment
    between two LiDAR point clouds.
    """

    def __init__(self, image_height: int,
                 image_width: int,
                 geom_b_min: float,
                 geom_b_max: float,
                 geom_b_ratio: float,
                 geom_rho_kernel: float,
                 geom_iterations: int,
                 photo_omega_depth: float,
                 photo_depth_rejection_threshold: float,
                 photo_rho_kernel: float,
                 photo_min_depth: float,
                 photo_max_depth: float,
                 photo_iterations: int):
        """
        Initializes the GSAligner class with the given parameters.

        Args:
            height (int): image height.
            width (int): image width.
            geom_b_min (float): The minimum geometric boundary.
            geom_b_max (float): The maximum geometric boundary.
            geom_b_ratio (float): The geometric boundary ratio.
            geom_rho_kernel (float): The geometric kernel density.
            geom_iterations (int): The number of geometric iterations.
            photo_omega_depth (float): The photometric omega depth.
            photo_depth_rejection_threshold (float): The threshold for photometric depth rejection.
            photo_rho_kernel (float): The photometric kernel density.
            photo_min_depth (float): minimum valid depth.
            photo_max_depth (float): maximum valid depth.
            photo_iterations (int): The number of photometric iterations.
        """

        self.impl = GSAlignerImpl(image_height, image_width,
                                  geom_b_min, geom_b_max, geom_b_ratio,
                                  geom_rho_kernel, geom_iterations,
                                  photo_omega_depth, photo_depth_rejection_threshold,
                                  photo_rho_kernel, photo_min_depth, photo_max_depth,
                                  photo_iterations)

    def set_reference(self, depth_image: torch.Tensor,
                      cloud: torch.Tensor,
                      projmat: torch.Tensor) -> None:
        """Set the reference frame for the alignment.

        Args:
            depth_image (torch.Tensor): (1,H,W) torch.float32 tensor
            cloud (torch.Tensor: (N, 3) torch.float32 tensor
            projmat (torch.Tensor): (4, 4) torch.float32 tensor **(MUST BE TRANSPOSED)**
        """

        self.impl.setReference(depth_image, cloud.cpu().double(), projmat)

    def set_query(self, depth_image: torch.Tensor,
                  cloud: torch.Tensor,
                  projmat: torch.Tensor) -> None:
        """Set the query frame for the alignment.

        Args:
            depth_image (torch.Tensor): (1,H,W) torch.float32 tensor
            cloud (torch.Tensor): (N, 3) torch.float32 tensor
            projmat (torch.Tensor): (4, 4) torch.float32 tensor **(MUST BE TRANSPOSED)**
        """
        self.impl.setQuery(depth_image, cloud.cpu().double(), projmat)

    def align_geometric(self, init_guess: torch.Tensor) -> \
            Tuple[torch.Tensor, float, float]:
        """Estimates the homogeneous transformation that maps the query frame to
        the reference frame using the geometric alignment strategy.

        Args:
            init_guess (torch.Tensor): (4, 4) torch.float32 tensor 

        Returns:
            A tuple containing the estimated transformation matrix as a
            (4, 4) torch.float32 tensor, a fitness score and a inlier error metric.
        """
        # Since geometric alignment is carried in double precision and on CPU,
        # we move the tensors accordingly here
        res, fitness, err_inliers = self.impl.alignGeometric(init_guess.cpu().double())
        return (res.float().cuda(), fitness, err_inliers)

    def align_photometric(self, init_guess: torch.Tensor) -> \
            Tuple[torch.Tensor, float, float]:
        """Estimates the homogeneous transformation that maps the query frame to
        the reference frame using the photometric alignment strategy.

        Args:
            init_guess (torch.Tensor): (4, 4) torch.float32 tensor

        Returns:
            A tuple containing the estimated transformation matrix as a
            (4, 4) torch.float32 tensor, a fitness score and a inlier error metric.

        """

        res, fitness, err_inliers = self.impl.alignPhotometric(
            init_guess.float().cpu())
        return res, fitness, err_inliers

    def align(self, init_guess: torch.Tensor) -> \
            Tuple[torch.Tensor, float, float]:
        """Estimates the homogeneous transformation that maps the query frame to
        the reference frame using both geometric and photometric strategies.
        First, the rough alignment is done using the geometric strategy, then
        the photometric strategy is applied to refine the transformation.

        Args:
            init_guess (torch.Tensor): (4, 4) torch.float32 tensor

        Returns:
            A tuple containing the estimated transformation matrix as a
            (4, 4) torch.float32 tensor, a fitness score and a inlier error metric.

        """

        res_geom, _, _ = self.align_geometric(init_guess)
        res, fitness, err_inliers = self.align_photometric(res_geom)
        return res, fitness, err_inliers
