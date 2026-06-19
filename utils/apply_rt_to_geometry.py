#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a rigid transform to a mesh or point cloud."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--R",
        type=float,
        nargs=9,
        required=True,
        metavar=("r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22"),
        help="Row-major 3x3 rotation matrix.",
    )
    parser.add_argument(
        "--t",
        type=float,
        nargs=3,
        required=True,
        metavar=("tx", "ty", "tz"),
        help="Translation vector.",
    )
    return parser.parse_args()


def build_transform(rotation_values, translation_values):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_values, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_values, dtype=np.float64)
    return transform


def transform_and_save(input_path: Path, output_path: Path, transform: np.ndarray):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if mesh.has_vertices() and mesh.has_triangles():
        transformed = o3d.geometry.TriangleMesh(mesh)
        transformed.transform(transform)
        ok = o3d.io.write_triangle_mesh(str(output_path), transformed, write_ascii=False)
        if not ok:
            raise RuntimeError(f"Failed to write transformed mesh: {output_path}")
        print(f"[OK] Saved transformed triangle mesh: {output_path}")
        return

    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.has_points():
        transformed = o3d.geometry.PointCloud(pcd)
        transformed.transform(transform)
        ok = o3d.io.write_point_cloud(str(output_path), transformed, write_ascii=False)
        if not ok:
            raise RuntimeError(f"Failed to write transformed point cloud: {output_path}")
        print(f"[OK] Saved transformed point cloud: {output_path}")
        return

    raise RuntimeError(f"Could not read a non-empty mesh or point cloud: {input_path}")


def main():
    args = parse_args()
    transform = build_transform(args.R, args.t)
    print("Using transform:")
    print(transform)
    transform_and_save(args.input_path, args.output_path, transform)


if __name__ == "__main__":
    main()
