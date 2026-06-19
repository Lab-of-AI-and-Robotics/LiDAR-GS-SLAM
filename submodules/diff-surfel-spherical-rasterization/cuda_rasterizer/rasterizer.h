#pragma once

#include <functional>

namespace CudaRasterizer {
class Rasterizer {
public:
  static void markVisible(int P, float *means3D, float *viewmatrix,
                          float *projmatrix, bool *present);
  static int forward(std::function<char *(size_t)> geometryBuffer,
                     std::function<char *(size_t)> binningBuffer,
                     std::function<char *(size_t)> imageBuffer, const int P,
                     const int width, const int height, const float *means3D,
                     const float *opacities, const float *scales,
                     const float scale_modifier, const float *rotations,
                     const float *transMat_precomp, const float *viewmatrix,
                     const float *projmatrix, const bool prefiltered,
                     float *out_others, int *radii = nullptr,
                     bool debug = false);

  static void backward(const int P, int R, const int width, int height,
                       const float *means3D, const float *scales,
                       const float scale_modifier, const float *rotations,
                       const float *transMat_precomp, const float *viewmatrix,
                       const float *projmatrix, const int *radii,
                       char *geom_buffer, char *binning_buffer,
                       char *image_buffer, const float *dL_depths,
                       float *dL_dmean2D, float *dL_dnormal, float *dL_dopacity,
                       float *dL_dmean3D, float *dL_dtransMat, float *dL_dscale,
                       float *dL_drot, bool debug);
};
} // namespace CudaRasterizer
