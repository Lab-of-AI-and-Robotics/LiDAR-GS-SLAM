import numpy as np
import logging
import natsort
from pathlib import Path
from utils.config_utils import PointCloudReaderConfig
from utils.enums import PointCloudReaderType
import open3d as o3d
from rosbags.highlevel import AnyReader
import sys
from typing import Iterable, List, Optional
import re
from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as PointCloud2
from rosbags.typesys.types import sensor_msgs__msg__PointField as PointField

logger = logging.getLogger("PointCloudReader")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class PointCloudReader():
    def __init__(self, config: PointCloudReaderConfig):
        self.n_clouds = 0
        self.current_index = 0

    def __len__(self):
        return self.n_clouds

    def __iter__(self):
        return self
        
    def __next__(self):
        raise NotImplementedError
    
    def get_timestamp(self, idx_or_filename):
        return 0.0

class PointCloudReader_Collections(PointCloudReader):
    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader.__init__(self, config)
        self.timestamps = None
        
        # Determine timestamp strategy
        if config.timestamp_filename is not None and Path(config.timestamp_filename).exists():
            self.timestamps = read_timestamps(Path(config.timestamp_filename))
            self.timestamp_mode = "file"
        elif config.timestamp_from_filename:
            self.timestamp_mode = "filename"
        else:
            self.timestamp_mode = "default"

    def get_timestamp(self, filename):
        if self.timestamp_mode == "file":
            # Assuming files are accessed sequentially matching timestamp list
            # We can't easily map filename -> index without full search unless we store map
            # Fallback: use current index if called during iteration, or find index
            if hasattr(self, 'filenames'):
                try:
                    idx = self.filenames.index(filename)
                    return self.timestamps[idx]
                except ValueError:
                    return 0.0
            return 0.0
        elif self.timestamp_mode == "filename":
            return str_to_timestamp(filename.stem)
        else:
            return 0.0

    def __next__(self):
        if self.current_index >= self.n_clouds:
            raise StopIteration
        
        filename = self.filenames[self.current_index]
        cloud = self.read_cloud(filename)
        timestamp = self.get_timestamp(filename)
        
        self.current_index += 1
        return cloud, timestamp

    def read_cloud(self, filename: Path):
        raise NotImplementedError

class PointCloudReader_BIN(PointCloudReader_Collections):
    def __init__(self, config: PointCloudReaderConfig):
        folder = Path(config.cloud_folder)
        config.cloud_folder = str(folder)
        
        PointCloudReader_Collections.__init__(self, config)
        self.filenames = sorted(Path(config.cloud_folder).glob("*.bin"))
        self.n_clouds = len(self.filenames)
        self.bin_format = config.bin_format if config.bin_format else "<f4"

    def read_cloud(self, filename: Path) -> np.ndarray:
        # print(f"[DEBUG] Reading cloud: {filename}")
        cloud_xyzi = np.fromfile(filename, self.bin_format).reshape(-1, 4)
        return cloud_xyzi

class PointCloudReader_PLY(PointCloudReader_Collections):
    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader_Collections.__init__(self, config)
        self.filenames = sorted(Path(config.cloud_folder).glob("*.ply"))
        self.n_clouds = len(self.filenames)

    def read_cloud(self, filename: Path) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(str(filename))
        points = np.asarray(pcd.points).astype(np.float32)
        return points

class PointCloudReader_PCD(PointCloudReader_Collections):
    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader_Collections.__init__(self, config)
        self.filenames = sorted(Path(config.cloud_folder).glob("*.pcd"))
        self.n_clouds = len(self.filenames)

    def read_cloud(self, filename: Path) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(str(filename))
        points = np.asarray(pcd.points).astype(np.float32)
        return points

class PointCloudReader_ROSBAG(PointCloudReader):
    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader.__init__(self, config)
        self.bag = None
        path = Path(config.cloud_folder)
        if path.is_file():
            self.bag = AnyReader([path])
        else:
            bag_filenames = natsort.natsorted([bag for bag in list(path.glob("*.bag"))])
            self.bag = AnyReader(bag_filenames)

        self.bag.open()
        
        connections = [x for x in self.bag.connections if x.topic == config.rosbag_topic]
        if len(connections) == 0:
            avail = {x.topic for x in self.bag.connections}
            logger.error(f"Topic {config.rosbag_topic} not found in {avail}")
            raise RuntimeError(f"Topic not found")
            
        self.n_clouds = self.bag.topics[config.rosbag_topic].msgcount
        self.cloud_loader = self.bag.messages(connections=connections)

        # [ADD] Timestamp Caching Logic
        # Bag file path logic handles list of bags, but here self.bag is AnyReader
        # We assume standard case with one main bag or folder. 
        # We'll use the cloud_folder path to store the cache.
        cache_path = Path(config.cloud_folder) / f"{Path(config.cloud_folder).name}_{config.rosbag_topic.replace('/', '_')}_timestamps.npy"
        
        self.timestamps = []
        
        if cache_path.exists():
            logger.info(f"Loading timestamps from cache: {cache_path}")
            try:
                self.timestamps = np.load(cache_path).tolist()
                logger.info(f"Loaded {len(self.timestamps)} timestamps.")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-scanning...")
                cache_path.unlink(missing_ok=True) # Delete corrupted cache
        
        if not self.timestamps:
            logger.info(f"Scanning bag timestamps for topic {config.rosbag_topic}...")
            scan_loader = self.bag.messages(connections=connections)
            count = 0
            for conn, ts, raw in scan_loader:
                self.timestamps.append(float(ts) / 1e9)
                count += 1
                if count % 2000 == 0:
                     logger.info(f"Scanning... found {count} frames so far")
            
            logger.info(f"Scanned {len(self.timestamps)} frames. Saving cache to {cache_path}")
            try:
                np.save(cache_path, np.array(self.timestamps))
            except Exception as e:
                logger.warning(f"Failed to save timestamp cache: {e}")

    def get_timestamp(self, idx_or_filename):
        if isinstance(idx_or_filename, int):
            if 0 <= idx_or_filename < len(self.timestamps):
                return self.timestamps[idx_or_filename]
        return 0.0

    def __next__(self):
        try:
            conn, _, raw = next(self.cloud_loader)
        except StopIteration:
            raise StopIteration
            
        cloud_msg = self.bag.deserialize(raw, conn.msgtype)
        timestamp = float(cloud_msg.header.stamp.sec) + float(cloud_msg.header.stamp.nanosec) / 1e9
        
        points_struct = read_points(cloud_msg)
        xyz = np.vstack([points_struct["x"], points_struct["y"], points_struct["z"]]).T
        
        
        # if "intensity" in points_struct.dtype.names:
        #     intensity = points_struct["intensity"]
        #     return np.hstack((xyz, intensity.reshape(-1, 1))), timestamp
        # elif "i" in points_struct.dtype.names:
        #     intensity = points_struct["i"]
        #     return np.hstack((xyz, intensity.reshape(-1, 1))), timestamp
        t_field = next((k for k in ("t", "ts", "time", "timestamp", "timestamps")
                if k in points_struct.dtype.names), None)

        if t_field is None:
            points_ts = None
        else:
            ts = points_struct[t_field].astype(np.float64)
            valid = np.isfinite(ts)
            if valid.sum() < 2:
                points_ts = None
            else:
                lo, hi = ts[valid].min(), ts[valid].max()
                if hi <= lo:
                    points_ts = None
                else:
                    points_ts = np.zeros_like(ts, dtype=np.float32)
                    points_ts[valid] = (ts[valid] - lo) / (hi - lo)

        return xyz, points_ts, timestamp

pointcloud_reader_available = {
    PointCloudReaderType.bin: PointCloudReader_BIN,
    PointCloudReaderType.ply: PointCloudReader_PLY,
    PointCloudReaderType.pcd: PointCloudReader_PCD,
    PointCloudReaderType.rosbag: PointCloudReader_ROSBAG
}

# --- Utils ---
def str_to_timestamp(timestamp_str: str) -> float:
    num_str = re.findall(r'\d+', timestamp_str)
    if len(num_str) == 1:
        return float(num_str[0])
    elif len(num_str) == 2:
        return float(num_str[0]) + float(num_str[1]) / 1e9
    else:
        raise ValueError(f"Invalid timestamp {timestamp_str}")

def read_timestamps(filename: Path) -> List[float]:
    with open(filename, "r") as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines]

# --- ROS Message Parsing ---
_DATATYPES = {}
_DATATYPES[PointField.INT8] = np.dtype(np.int8)
_DATATYPES[PointField.UINT8] = np.dtype(np.uint8)
_DATATYPES[PointField.INT16] = np.dtype(np.int16)
_DATATYPES[PointField.UINT16] = np.dtype(np.uint16)
_DATATYPES[PointField.INT32] = np.dtype(np.int32)
_DATATYPES[PointField.UINT32] = np.dtype(np.uint32)
_DATATYPES[PointField.FLOAT32] = np.dtype(np.float32)
_DATATYPES[PointField.FLOAT64] = np.dtype(np.float64)
DUMMY_FIELD_PREFIX = "unnamed_field"

def read_points(cloud: PointCloud2, field_names: Optional[List[str]] = None, uvs: Optional[Iterable] = None, reshape_organized_cloud: bool = False) -> np.ndarray:
    points = np.ndarray(
        shape=(cloud.width * cloud.height,),
        dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=cloud.data,
    )
    if field_names is not None:
        points = points[list(field_names)]
    if bool(sys.byteorder != "little") != bool(cloud.is_bigendian):
        points = points.byteswap(inplace=True)
    if uvs is not None:
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
        points = points[uvs]
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)
    return points

def dtype_from_fields(fields: Iterable[PointField], point_step: Optional[int] = None) -> np.dtype:
    field_names = []
    field_offsets = []
    field_datatypes = []
    for i, field in enumerate(fields):
        datatype = _DATATYPES[field.datatype]
        if field.name == "":
            name = f"{DUMMY_FIELD_PREFIX}_{i}"
        else:
            name = field.name
        for a in range(field.count):
            if field.count > 1:
                subfield_name = f"{name}_{a}"
            else:
                subfield_name = name
            field_names.append(subfield_name)
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)
    dtype_dict = {"names": field_names, "formats": field_datatypes, "offsets": field_offsets}
    if point_step is not None:
        dtype_dict["itemsize"] = point_step
    return np.dtype(dtype_dict)