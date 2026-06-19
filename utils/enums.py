from enum import Enum

class DatasetType(str, Enum):
    generic = "generic"
    vbr = "vbr"
    kitti = "kitti"
    ncd = "ncd"
    oxspires = "oxspires"
    oxspires_vilens = "oxspires_vilens"

class PointCloudReaderType(str, Enum):
    bin = "bin"
    ply = "ply"
    pcd = "pcd"
    rosbag = "rosbag"
    null = "null"

class TrajectoryReaderType(str, Enum):
    kitti = "kitti"
    tum = "tum"
    vilens = "vilens"
    null = "null"

class TrajectoryWriterType(str, Enum):
    kitti = "kitti"
    tum = "tum"
