from pytransform3d.rotations import (
    quaternion_from_matrix,
    quaternion_wxyz_from_xyzw,
    norm_matrix
)
import re
from utils.pointcloud_utils import read_timestamps
from pytransform3d.transformations import (transform_from_pq, check_transform)
import numpy as np
from pathlib import Path
from typing import List
from utils.config_utils import (
    TrajectoryReaderType,
    TrajectoryWriterType,
    TrajectoryReaderConfig
)

class TrajectoryReader:
    def __init__(self, config: TrajectoryReaderConfig):
        self.dtol = config.timestamp_dtol
        self.timestamps = []
        self.poses = []
        self.current_index = 0
        
        self.gt_T_s = np.eye(4)
        
        if config.gt_T_sensor_t_xyz_q_xyzw is not None:
            # Expected input: [tx, ty, tz, qx, qy, qz, qw]
            # pytransform3d uses [w, x, y, z] for quat usually or we adjust.
            # Splat-LOAM code uses: quaternion_wxyz_from_xyzw
            gt_T_s_pq = np.array(config.gt_T_sensor_t_xyz_q_xyzw, dtype=np.float32)
            # transform_from_pq expects [tx, ty, tz, qw, qx, qy, qz]
            # Convert xyzw -> wxyz
            gt_T_s_pq[3:] = quaternion_wxyz_from_xyzw(gt_T_s_pq[3:])
            self.gt_T_s = transform_from_pq(gt_T_s_pq)
            
        elif config.gt_T_sensor_kitti_filename is not None:
            fpath = Path(config.gt_T_sensor_kitti_filename)
            if fpath.exists():
                with open(fpath) as f:
                    for line in f.readlines():
                        if "Tr:" in line: # raw kitti calib
                            line = line[3:]
                            pose_vect = np.array([float(x) for x in line.split()])
                            pose = pose_vect.reshape(3, 4)
                            pose = np.vstack((pose, [0, 0, 0, 1]))
                            self.gt_T_s = pose
                        elif "Tr_velo_to_cam:" in line: # odometry calib
                             line = line.split(":")[-1]
                             pose_vect = np.array([float(x) for x in line.split()])
                             pose = pose_vect.reshape(3, 4)
                             pose = np.vstack((pose, [0, 0, 0, 1]))
                             self.gt_T_s = pose
        
    def __call__(self, timestamp: float, *args, **kwargs) -> np.ndarray:
        try:
            idx = self._find_closest_timestamp_idx(timestamp)
        except RuntimeError:
            # Fallback or re-raise
            raise
        return self.poses[idx] @ self.gt_T_s

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.current_index >= len(self.poses):
            raise StopIteration
        pose = self.poses[self.current_index] @ self.gt_T_s
        self.current_index += 1
        return pose

    @staticmethod
    def _parse_numeric_row(line: str):
        """
        Parse one trajectory row supporting both whitespace and CSV delimiters.
        Non-numeric rows (e.g. headers like 'sec,nsec,...') are skipped.
        """
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        # Remove inline comments if present.
        line = line.split("#", 1)[0].strip()
        if not line:
            return None

        tokens = [t for t in re.split(r"[\s,]+", line) if t]
        if not tokens:
            return None

        try:
            return np.array([float(x) for x in tokens], dtype=np.float64)
        except ValueError:
            return None

    @staticmethod
    def _decode_timestamp_and_pose(row: np.ndarray):
        """
        Supported layouts:
        - 8 cols : [t, x, y, z, qx, qy, qz, qw]
        - 9 cols : [sec, nsec, x, y, z, qx, qy, qz, qw]
        - 10+ cols : [counter, sec, nsec, x, y, z, qx, qy, qz, qw, ...]
        """
        if row is None:
            return None, None
        if row.size == 8:
            return row[0].item(), row[1:8].copy()
        if row.size == 9:
            ts = row[0].item() + row[1].item() / 1e9
            return ts, row[2:9].copy()
        if row.size >= 10:
            ts = row[1].item() + row[2].item() / 1e9
            return ts, row[3:10].copy()
        return None, None
        
    def get_pose(self, idx):
         if idx < len(self.poses):
             return self.poses[idx] @ self.gt_T_s
         return np.eye(4)

    def _find_closest_timestamp_idx(self, timestamp: float) -> int:
        if not self.timestamps:
            raise RuntimeError("No timestamps loaded")
            
        # Optimize using binary search (O(log N)) assuming timestamps are sorted
        # Find insertion point
        idx = np.searchsorted(self.timestamps, timestamp)
        
        # Check candidates: idx (right) and idx-1 (left)
        candidates = []
        if idx < len(self.timestamps):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        if not candidates:
            raise RuntimeError(f"Could not find timestamp for {timestamp}")
            
        # Find closest among candidates
        closest_idx = min(candidates, key=lambda i: abs(self.timestamps[i] - timestamp))
        
        if abs(self.timestamps[closest_idx] - timestamp) > self.dtol:
             raise RuntimeError(f"No timestamp found within tolerance {self.dtol} for {timestamp}")
        return closest_idx
    # Legacy closest-timestamp search.
    #def _find_closest_timestamp_idx(self, timestamp: float) -> int:
        #if not self.timestamps:
        #    raise RuntimeError("No timestamps loaded")
        # Simple nearest search
        #closest_idx = min(
        #    range(len(self.timestamps)),
        #    key=lambda i: abs(self.timestamps[i] - timestamp)
        #)
        #if abs(self.timestamps[closest_idx] - timestamp) > self.dtol:
        #     raise RuntimeError(f"No timestamp found within tolerance {self.dtol} for {timestamp}")
        #return closest_idx

class TrajectoryReader_KITTI(TrajectoryReader):
    def __init__(self, config: TrajectoryReaderConfig):
        TrajectoryReader.__init__(self, config)
        if config.filename and Path(config.filename).exists():
            print(f"[TrajectoryReader] Loading poses from {config.filename}")
            with open(config.filename, "r") as f:
                lines = f.readlines()
            for line in lines:
                pose_vect = np.array([float(x) for x in line.split()])
                pose = pose_vect.reshape(3, 4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                self.poses.append(pose)
            print(f"[TrajectoryReader] Loaded {len(self.poses)} poses.")
                
        if config.timestamp_from_filename_kitti is not None:
             ts_path = Path(config.timestamp_from_filename_kitti)
             if ts_path.exists():
                 self.timestamps = read_timestamps(ts_path)

class TrajectoryReader_TUM(TrajectoryReader):
    def __init__(self, config: TrajectoryReaderConfig):
        TrajectoryReader.__init__(self, config)
        if config.filename and Path(config.filename).exists():
            with open(config.filename, "r") as f:
                lines = f.readlines()
            for line in lines:
                row = self._parse_numeric_row(line)
                ts, pq = self._decode_timestamp_and_pose(row)
                if ts is None or pq is None or pq.size != 7:
                    continue
                self.timestamps.append(ts)
                # Convert xyzw to wxyz for pytransform3d
                pq[3:] = quaternion_wxyz_from_xyzw(pq[3:])
                self.poses.append(transform_from_pq(pq))

class TrajectoryReader_VILENS(TrajectoryReader):
    def __init__(self, config: TrajectoryReaderConfig):
        TrajectoryReader.__init__(self, config)
        if config.filename and Path(config.filename).exists():
             with open(config.filename, "r") as f:
                lines = f.readlines()
             for line in lines:
                row = self._parse_numeric_row(line)
                ts, pq = self._decode_timestamp_and_pose(row)
                if ts is None or pq is None or pq.size != 7:
                    continue
                self.timestamps.append(ts)
                pq[3:] = quaternion_wxyz_from_xyzw(pq[3:])
                self.poses.append(transform_from_pq(pq))

class TrajectoryReader_NULL(TrajectoryReader):
    def __init__(self, config: TrajectoryReaderConfig):
        TrajectoryReader.__init__(self, config)
        
    def __call__(self, _: float, *args, **kwargs) -> np.ndarray:
        return np.eye(4)
    def __next__(self):
        return np.eye(4)
    def get_pose(self, idx):
        return np.eye(4)

trajectory_reader_available = {
    TrajectoryReaderType.kitti: TrajectoryReader_KITTI,
    TrajectoryReaderType.tum: TrajectoryReader_TUM,
    TrajectoryReaderType.vilens: TrajectoryReader_VILENS,
    TrajectoryReaderType.null: TrajectoryReader_NULL
}
