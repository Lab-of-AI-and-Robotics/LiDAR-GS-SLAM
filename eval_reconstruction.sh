#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash eval_reconstruction.sh \
    --result-dir <results/YYYY-MM-DD_HH-MM-SS> \
    --gt-traj <groundtruth trajectory> \
    --gt-map <groundtruth mesh/pointcloud ply> \
    [--traj-format tum|kitti] \
    [--t-max-diff 0.03]

Pipeline:
  1. Build mesh with Splat-LOAM
  2. Align estimated trajectory to GT with evo_ape
  3. Apply evo R/t to the estimated mesh
  4. Crop GT map with the aligned mesh
  5. Evaluate reconstruction with Splat-LOAM eval_recon
EOF
}

RESULT_DIR=""
GT_TRAJ=""
GT_MAP=""
TRAJ_FORMAT="tum"
T_MAX_DIFF="0.03"
MESH_SAMPLE_POINTS="10000000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --gt-traj)
      GT_TRAJ="$2"
      shift 2
      ;;
    --gt-map|--gt-mesh|--gt-pcd)
      GT_MAP="$2"
      shift 2
      ;;
    --traj-format)
      TRAJ_FORMAT="$2"
      shift 2
      ;;
    --t-max-diff)
      T_MAX_DIFF="$2"
      shift 2
      ;;
    --mesh-sample-points)
      MESH_SAMPLE_POINTS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RESULT_DIR" || -z "$GT_TRAJ" || -z "$GT_MAP" ]]; then
  echo "[ERROR] --result-dir, --gt-traj, and --gt-map are required."
  usage
  exit 1
fi

if [[ ! -d "$RESULT_DIR" ]]; then
  echo "[ERROR] Result directory not found: $RESULT_DIR"
  exit 1
fi

if [[ ! -f "$GT_TRAJ" ]]; then
  echo "[ERROR] GT trajectory not found: $GT_TRAJ"
  exit 1
fi

if [[ ! -f "$GT_MAP" ]]; then
  echo "[ERROR] GT map not found: $GT_MAP"
  exit 1
fi

case "$TRAJ_FORMAT" in
  tum)
    EST_TRAJ="$RESULT_DIR/est_traj_tum.txt"
    ;;
  kitti)
    EST_TRAJ="$RESULT_DIR/est_traj_kitti.txt"
    ;;
  *)
    echo "[ERROR] Unsupported --traj-format: $TRAJ_FORMAT"
    echo "Supported formats: tum, kitti"
    exit 1
    ;;
esac

if [[ ! -f "$EST_TRAJ" ]]; then
  echo "[ERROR] Estimated trajectory not found: $EST_TRAJ"
  echo "Run SLAM to normal completion first so trajectory exports are created."
  exit 1
fi

MESH_OUT="$RESULT_DIR/mesh.ply"
EVO_LOG="$RESULT_DIR/evo_ape_align.log"
MESH_ALIGNED="${MESH_OUT%.ply}-gt-align.ply"
GT_CROP="$RESULT_DIR/gt_crop.ply"
EVAL_CSV="$RESULT_DIR/eval_recon.csv"

echo "[1/5] Create mesh"
python3 Splat-LOAM/run.py mesh "$RESULT_DIR" --output-filename "$MESH_OUT"

if [[ ! -f "$MESH_OUT" ]]; then
  echo "[ERROR] Mesh file not found: $MESH_OUT"
  exit 1
fi

echo "[2/5] Align trajectory with evo_ape"
if [[ "$TRAJ_FORMAT" == "tum" ]]; then
  evo_ape tum "$GT_TRAJ" "$EST_TRAJ" --align --t_max_diff "$T_MAX_DIFF" --plot_mode xy -v \
    2>&1 | tee "$EVO_LOG"
else
  evo_ape kitti "$GT_TRAJ" "$EST_TRAJ" --align --plot_mode xy -v \
    2>&1 | tee "$EVO_LOG"
fi

echo "[3/5] Parse evo R/t and align mesh"
mapfile -t RT_LINES < <(
python3 - "$EVO_LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")

r_match = re.search(r"Rotation of alignment:\s*\n\[\[(.*?)\]\]", text, re.S)
if not r_match:
    raise SystemExit("Could not parse 'Rotation of alignment' from evo log.")

r_nums = [
    float(x)
    for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", r_match.group(1))
]
if len(r_nums) != 9:
    raise SystemExit(f"Expected 9 rotation values, got {len(r_nums)}")

t_match = re.search(r"Translation of alignment:\s*\[([^\]]+)\]", text)
if not t_match:
    raise SystemExit("Could not parse 'Translation of alignment' from evo log.")

t_nums = [
    float(x)
    for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t_match.group(1))
]
if len(t_nums) != 3:
    raise SystemExit(f"Expected 3 translation values, got {len(t_nums)}")

print(" ".join(f"{v:.12f}" for v in r_nums))
print(" ".join(f"{v:.12f}" for v in t_nums))
PY
)

if [[ "${#RT_LINES[@]}" -ne 2 ]]; then
  echo "[ERROR] Failed to parse R/t from evo log: $EVO_LOG"
  exit 1
fi

read -r -a R_ARR <<< "${RT_LINES[0]}"
read -r -a T_ARR <<< "${RT_LINES[1]}"

python3 utils/apply_rt_to_geometry.py "$MESH_OUT" "$MESH_ALIGNED" \
  --R "${R_ARR[@]}" \
  --t "${T_ARR[@]}"

echo "[4/5] Crop GT map with aligned mesh"
python3 Splat-LOAM/run.py crop_recon "$GT_MAP" "$MESH_ALIGNED" \
  --mesh-sample-point "$MESH_SAMPLE_POINTS" \
  --output-filename "$GT_CROP"

if [[ ! -f "$GT_CROP" ]]; then
  echo "[ERROR] Cropped GT file not found: $GT_CROP"
  exit 1
fi

echo "[5/5] Evaluate reconstruction"
python3 Splat-LOAM/run.py eval_recon "$GT_CROP" "$MESH_ALIGNED" \
  --mesh-sample-point "$MESH_SAMPLE_POINTS" \
  --output-filename "$EVAL_CSV"

echo "[DONE] Reconstruction evaluation complete."
echo "[INFO] Mesh:         $MESH_OUT"
echo "[INFO] Aligned mesh: $MESH_ALIGNED"
echo "[INFO] Cropped GT:   $GT_CROP"
echo "[INFO] Eval CSV:     $EVAL_CSV"
