#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Installing diff-surfel-spherical-rasterization"
pip install "$ROOT_DIR/submodules/diff-surfel-spherical-rasterization"

echo "[2/4] Installing simple-knn"
pip install "$ROOT_DIR/submodules/simple-knn"

echo "[3/4] Installing modified fast_gicp"
(
  cd "$ROOT_DIR/submodules/fast_gicp"
  python setup.py install
)

echo "[4/4] Building MapClosures pybind module"
cmake -S "$ROOT_DIR/submodules/MapClosures/python" \
  -B "$ROOT_DIR/submodules/MapClosures/python/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ROOT_DIR/submodules/MapClosures/python/map_closures/pybind"
cmake --build "$ROOT_DIR/submodules/MapClosures/python/build" -j
cmake --install "$ROOT_DIR/submodules/MapClosures/python/build"

echo "[DONE] Submodule extensions installed."
