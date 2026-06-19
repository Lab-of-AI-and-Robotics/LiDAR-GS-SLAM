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

#include <cuda.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD {
void render(const dim3 grid, dim3 block, const uint2 *ranges,
            const uint32_t *point_list, int W, int H, const float *projmatrix,
            const float2 *means2D, const float4 *normal_opacity,
            const float *transMats, const float *depths, const float *final_Ts,
            const uint32_t *n_contrib, const float *dL_depths,
            float *dL_dtransMat, float3 *dL_dmean2D, float *dL_dnormal3D,
            float *dL_dopacity);

void preprocess(int P, const float3 *means, const int *radii,
                const bool *clamped, const glm::vec2 *scales,
                const glm::vec4 *rotations, const float scale_modifier,
                const float *transMats, const float *view, const float *proj,
                const int W, const int H, float3 *dL_dmean2D,
                const float *dL_dnormal3D, float *dL_dtransMat,
                glm::vec3 *dL_dmeans, glm::vec2 *dL_dscale, glm::vec4 *dL_drot);
} // namespace BACKWARD
