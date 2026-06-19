#!/usr/bin/env bash
set -e

# Datasets to iterate over. Override by passing paths as CLI args.
DATASETS=(
  "/mnt/storage/Downloads/lidar_dataset/kitti/00" 
  "/mnt/storage/Downloads/lidar_dataset/kitti/01"
  "/mnt/storage/Downloads/lidar_dataset/kitti/02"
  "/mnt/storage/Downloads/lidar_dataset/kitti/03"
  "/mnt/storage/Downloads/lidar_dataset/kitti/04"
  "/mnt/storage/Downloads/lidar_dataset/kitti/05"
  "/mnt/storage/Downloads/lidar_dataset/kitti/06"
  "/mnt/storage/Downloads/lidar_dataset/kitti/07"
  "/mnt/storage/Downloads/lidar_dataset/kitti/08"
  "/mnt/storage/Downloads/lidar_dataset/kitti/09"
  "/mnt/storage/Downloads/lidar_dataset/kitti/10"
         # replace with your dataset root
)

if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
fi

for num in 1 2 3 4 5; do
  echo "=== Running dataset: ${num} ==="
  for ds in "${DATASETS[@]}"; do
    name=$(basename "$ds")
    echo "=== Running dataset: ${ds} ==="
    
    # loop_overlap_th=$(awk -v n="$num" 'BEGIN { printf "%.3f", 0.1 + 0.02*n }')
    loop_overlap_th=$(awk -v n="$num" 'BEGIN { printf "%.3f", 0.1*n }')

    out="./results/${name}/${num}_loop_overlap_${loop_overlap_th}"
    mkdir -p "$out"

      python3 gs_icp_slam.py \
        --dataset_path "$ds" \
        --config_file "./configs/kitti.yaml" \
        --output_path "$out" \
        --keyframe_th 0.8 \
        --overlapped_th 0.895 \
        --loop_overlap_th "$loop_overlap_th"\
        --use_dynamic_fov False \
        --n_trackable_keyframes 20
  done
done
