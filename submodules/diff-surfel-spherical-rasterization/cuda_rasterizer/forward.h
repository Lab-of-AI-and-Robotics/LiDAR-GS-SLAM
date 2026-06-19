/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include "auxiliary.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(int P, const float *orig_points, const glm::vec2 *scales,
                const float scale_modifier, const glm::vec4 *rotations,
                const float *opacities, bool *clamped,
                const float *transMat_precomp, const float *viewmatrix,
                const float *projmatrix, const int W, int H, int *radii,
                float2 *points_xy_image, float *depths, float *transMats,
                float4 *normal_opacity, const dim3 grid,
                uint32_t *tiles_touched, bool prefiltered);

// Main rasterization method.
void render(const dim3 grid, dim3 block, const uint2 *ranges,
            const uint32_t *point_list, int W, int H, const float *projmatrix,
            const float2 *points_xy_image, const float *transMats,
            const float *depths, const float4 *normal_opacity, float *final_T,
            uint32_t *n_contrib, float *out_others);

} // namespace FORWARD
