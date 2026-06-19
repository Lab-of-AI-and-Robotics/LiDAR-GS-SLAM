from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import OmegaConf
from pathlib import Path
from .enums import DatasetType, TrajectoryReaderType, PointCloudReaderType, TrajectoryWriterType

@dataclass
class TrajectoryReaderConfig:
    reader_type: Optional[TrajectoryReaderType] = None
    filename: Optional[str] = None
    timestamp_dtol: float = 20 # 20ms tolerance
    timestamp_from_filename_kitti: Optional[str] = None
    gt_T_sensor_t_xyz_q_xyzw: Optional[tuple] = None
    gt_T_sensor_kitti_filename: Optional[str] = None

@dataclass
class PointCloudReaderConfig:
    cloud_folder: str = ""
    cloud_format: Optional[PointCloudReaderType] = None
    timestamp_from_filename: Optional[bool] = False
    timestamp_filename: Optional[str] = None
    bin_format: Optional[str] = "<f4"
    rosbag_topic: Optional[str] = None

@dataclass
class DatasetConfig:
    dataset_type: DatasetType = DatasetType.kitti
    dataset_path: str = "dataset/path"
    
    # Composed configurations
    trajectory_reader: Optional[TrajectoryReaderConfig] = field(default_factory=TrajectoryReaderConfig)
    cloud_reader: Optional[PointCloudReaderConfig] = field(default_factory=PointCloudReaderConfig)
    
    # Flag to skip unsynced clouds
    skip_clouds_wno_sync: Optional[bool] = False
    
    # Additional calibration/depth fields if needed directly
    min_depth: float = 1.0
    max_depth: float = 80.0

@dataclass
class SLAMConfig:
    keyframe_th: float = 0.4
    knn_maxd: float = 99999.0
    overlapped_th: float = 0.005
    overlapped_th2: float = 0.005
    max_correspondence_distance: float = 1.0
    trackable_opacity_th: float = 0.01
    downsample_rate: int = 20
    downsample_voxel_size: float = 0.4
    loop_constraint_noise: float = 1e-6
    keyframe_freq: int = 100
    n_trackable_keyframes: int = 100
    use_dynamic_fov: bool = True
    loop_overlap_th : float = 0.7
  
    # densify related parameters
    densify_threshold_opacity: float = 0.2
    densify_percentage: float = 0.40
    opt_scaling_max: float = 0.1
    use_densify: bool = True
    densify_start_iteration: int = 100

    # pruning related parameters
    pruning_min_opacity: float = 0.05
    pruning_min_scale: float = 0.001
    loop_cooldown_time: float = 5.0
    # loss
    opt_lambda_dist: float = 0.01
    decay_speed: float = 0.0125
    
    # Intrinsics / LiDAR Specs
    W: int = 1800
    H: int = 64
    fov_x_deg: float = 60.0
    fov_y_deg: float = 26.8
    depth_scale: float = 1.0
    depth_trunc: float = 80.0
    densify_frequency: int = 50

@dataclass
class Configuration:
    data: DatasetConfig = field(default_factory=DatasetConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    
    output_path: str = "output/default"
    verbose: bool = False
    demo: bool = False
    test: Optional[str] = None
    
    inherit_from: Optional[str] = None

def load_configuration(filename: Path, cli_args: list = None) -> Configuration:
    default_cfg = OmegaConf.structured(Configuration)
    if filename and filename.exists():
        derived_cfg = OmegaConf.load(filename)
        if derived_cfg.get("inherit_from") is not None:
            base_cfg = load_configuration(Path(derived_cfg["inherit_from"]))
            cfg = OmegaConf.merge(default_cfg, base_cfg, derived_cfg)
        else:
            cfg = OmegaConf.merge(default_cfg, derived_cfg)
    else:
        cfg = default_cfg

    if cli_args is not None:
        override_cfg = OmegaConf.from_cli(cli_args)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg
