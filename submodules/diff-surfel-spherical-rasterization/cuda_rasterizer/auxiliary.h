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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "stdio.h"
#include <glm/glm.hpp>

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

#define RENDER_AXUTILITY 1
#define DEPTH_OFFSET 0
#define ALPHA_OFFSET 1
#define NORMAL_OFFSET 2
#define MIDDEPTH_OFFSET 5
#define DISTORTION_OFFSET 6

// distortion helper macros
#define BACKFACE_CULL 1
#define DUAL_VISIABLE 1
#define DETACH_WEIGHT 0

__device__ const float near_n = 0.1;
__device__ const float far_n = 120.0;
__device__ const float FilterInvSquare = 2.0f;

__forceinline__ __device__ float ndc2Pix(float v, int S) {
  return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius,
                                        uint2 &rect_min, uint2 &rect_max,
                                        dim3 grid) {
  rect_min = {min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
              min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))};
  rect_max = {
      min(grid.x,
          max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
      min(grid.y,
          max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ void getRectSpherical(const float2 p, int max_radius,
                                                 int2 &rect_min, int2 &rect_max,
                                                 dim3 grid) {
  rect_min = {
      ((int)(p.x - max_radius - BLOCK_X - 1) / BLOCK_X),
      min((int)grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))};
  rect_max = {
      ((int)(p.x + max_radius + BLOCK_X - 1) / BLOCK_X),
      min((int)grid.y,
          max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3 &p,
                                                    const float *matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
  };
  return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3 &p,
                                                    const float *matrix) {
  float4 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
      matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]};
  return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3 &p,
                                                  const float *matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
      matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
      matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float3
transformVec4x3Transpose(const float3 &p, const float *matrix) {
  float3 transformed = {
      matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
      matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
      matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
  };
  return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv) {
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
  float dnormvdz =
      (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
      invsum32;
  return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv) {
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float3 dnormvdv;
  dnormvdv.x =
      ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) *
      invsum32;
  dnormvdv.y =
      (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) *
      invsum32;
  dnormvdv.z =
      (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) *
      invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv) {
  float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

  float4 vdv = {v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w};
  float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
  float4 dnormvdv;
  dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
  dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
  dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
  dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
  return dnormvdv;
}

__forceinline__ __device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__forceinline__ __device__ float3 operator*(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float2 operator*(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

__forceinline__ __device__ float3 operator*(float f, float3 a) {
  return make_float3(f * a.x, f * a.y, f * a.z);
}

__forceinline__ __device__ float2 operator*(float f, float2 a) {
  return make_float2(f * a.x, f * a.y);
}

__forceinline__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__forceinline__ __device__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __device__ float sumf3(float3 a) { return a.x + a.y + a.z; }

__forceinline__ __device__ float sumf2(float2 a) { return a.x + a.y; }

__forceinline__ __device__ float3 sqrtf3(float3 a) {
  return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
}

__forceinline__ __device__ float2 sqrtf2(float2 a) {
  return make_float2(sqrtf(a.x), sqrtf(a.y));
}

__forceinline__ __device__ float3 minf3(float f, float3 a) {
  return make_float3(min(f, a.x), min(f, a.y), min(f, a.z));
}

__forceinline__ __device__ float2 minf2(float f, float2 a) {
  return make_float2(min(f, a.x), min(f, a.y));
}

__forceinline__ __device__ float3 maxf3(float f, float3 a) {
  return make_float3(max(f, a.x), max(f, a.y), max(f, a.z));
}

__forceinline__ __device__ float2 maxf2(float f, float2 a) {
  return make_float2(max(f, a.x), max(f, a.y));
}

__forceinline__ __device__ float2 clampf2(float2 a, float2 _min, float2 _max) {
  return make_float2(max(min(a.x, _max.x), _min.x),
                     max(min(a.y, _max.y), _min.y));
}

__forceinline__ __device__ float normf3(float3 a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__forceinline__ __device__ float normf2(float2 a) {
  return sqrtf(a.x * a.x + a.y * a.y);
}

__forceinline__ __device__ float3 operator*(float4 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float dotf4x3(float4 a, float3 b) {
  return sumf3(a * b);
}

__forceinline__ __device__ float4 operator*(float f, float4 a) {
  return make_float4(f * a.x, f * a.y, f * a.z, f * a.w);
}

__forceinline__ __device__ float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __device__ float3 tofloat3(float4 a) {
  return make_float3(a.x, a.y, a.z);
}

__forceinline__ __device__ float dot3f(float3 a, float3 b) {
  return sumf3(a * b);
}

/**
 * @brief LiDAR frustum check is done by checking p_view's norm
 *
 * @param idx
 * @param orig_points
 * @param viewmatrix
 * @param projmatrix
 * @param prefiltered
 * @param p_view
 * @return __forceinline__
 */
__forceinline__ __device__ bool in_frustum(int idx, const float *orig_points,
                                           const float *viewmatrix,
                                           const float *projmatrix,
                                           bool prefiltered, float3 &p_view) {
  float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1],
                   orig_points[3 * idx + 2]};

  p_view = transformPoint4x3(p_orig, viewmatrix);
  if (normf3(p_view) <= near_n or normf3(p_view) > far_n) {
    if (prefiltered) {
      printf("Point is filtered although prefiltered is set. This shouldn't "
             "happen!");
      __trap();
    }
    return false;
  }
  return true;
}

// adopt from gsplat:
// https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/forward.cu
inline __device__ glm::mat3 quat_to_rotmat(const glm::vec4 quat) {
  // quat to rotation matrix
  float s = rsqrtf(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y +
                   quat.z * quat.z);
  float w = quat.x * s;
  float x = quat.y * s;
  float y = quat.z * s;
  float z = quat.w * s;

  // glm matrices are column-major
  return glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z),
                   2.f * (x * z - w * y), 2.f * (x * y - w * z),
                   1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
                   2.f * (x * z + w * y), 2.f * (y * z - w * x),
                   1.f - 2.f * (x * x + y * y));
}

inline __device__ glm::vec4 quat_to_rotmat_vjp(const glm::vec4 quat,
                                               const glm::mat3 v_R) {
  float s = rsqrtf(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y +
                   quat.z * quat.z);
  float w = quat.x * s;
  float x = quat.y * s;
  float y = quat.z * s;
  float z = quat.w * s;

  glm::vec4 v_quat;
  // v_R is COLUMN MAJOR
  // w element stored in x field
  v_quat.x =
      2.f * (
                // v_quat.w = 2.f * (
                x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                z * (v_R[0][1] - v_R[1][0]));
  // x element in y field
  v_quat.y =
      2.f *
      (
          // v_quat.x = 2.f * (
          -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
          z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1]));
  // y element in z field
  v_quat.z =
      2.f *
      (
          // v_quat.y = 2.f * (
          x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
          z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2]));
  // z element in w field
  v_quat.w =
      2.f *
      (
          // v_quat.z = 2.f * (
          x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
          2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]));
  return v_quat;
}

inline __device__ glm::mat3 scale_to_mat(const glm::vec2 scale,
                                         const float glob_scale) {
  glm::mat3 S = glm::mat3(1.f);
  S[0][0] = glob_scale * scale.x;
  S[1][1] = glob_scale * scale.y;
  // S[2][2] = glob_scale * scale.z;
  return S;
}

inline __device__ glm::vec2 xyz_to_sph(const glm::vec3 &p) {
  return glm::vec2(atan2f(p[1], p[0]),
                   atan2f(p[2], sqrtf(p[0] * p[0] + p[1] * p[1])));
}

inline __device__ glm::vec3 sph_to_xyz(const glm::vec2 &p) {
  return glm::vec3(cosf(p[0]) * cosf(p[1]), sinf(p[0]) * cosf(p[1]),
                   sinf(p[1]));
}

inline __device__ float3 sph_to_xyz(const float2 &p) {
  float s_p_x, c_p_x;
  float s_p_y, c_p_y;
  __sincosf(p.x, &s_p_x, &c_p_x);
  __sincosf(p.y, &s_p_y, &c_p_y);
  return make_float3(c_p_x * c_p_y, s_p_x * c_p_y, s_p_y);
}

#define CHECK_CUDA(A, debug)                                                   \
  A;                                                                           \
  if (debug) {                                                                 \
    auto ret = cudaDeviceSynchronize();                                        \
    if (ret != cudaSuccess) {                                                  \
      std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__   \
                << ": " << cudaGetErrorString(ret);                            \
      throw std::runtime_error(cudaGetErrorString(ret));                       \
    }                                                                          \
  }

#endif
