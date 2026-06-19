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

#include "auxiliary.h"
#include "backward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderCUDA(const uint2 *__restrict__ ranges,
               const uint32_t *__restrict__ point_list, int W, int H,
               const float *__restrict__ projmatrix,
               const float2 *__restrict__ points_xy_image,
               const float4 *__restrict__ normal_opacity,
               const float *__restrict__ transMats, const float *__restrict__,
               const float *__restrict__ final_Ts,
               const uint32_t *__restrict__ n_contrib,
               const float *__restrict__ dL_depths,
               float *__restrict__ dL_dtransMat, float3 *__restrict__,
               float *__restrict__ dL_dnormal3D,
               float *__restrict__ dL_dopacity) {
  // We rasterize again. Compute necessary block info.
  auto block = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min = {block.group_index().x * BLOCK_X,
                         block.group_index().y * BLOCK_Y};
  const uint2 pix = {pix_min.x + block.thread_index().x,
                     pix_min.y + block.thread_index().y};
  const uint32_t pix_id = W * pix.y + pix.x;
  const float2 pixf = {(float)pix.x, (float)pix.y};

  const bool inside =
      pix.x < static_cast<float>(W) && pix.y < static_cast<float>(H);
  const uint2 range =
      ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  bool done = !inside;
  int toDo = range.y - range.x;

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_normal_opacity[BLOCK_SIZE];
  __shared__ float4 collected_Tu[BLOCK_SIZE];
  __shared__ float4 collected_Tv[BLOCK_SIZE];
  __shared__ float4 collected_Tw[BLOCK_SIZE];

  // In the forward, we stored the final value for T, the
  // product of all (1 - alpha) factors.
  const float T_final = inside ? final_Ts[pix_id] : 0;
  float T = T_final;

  // We start from the back. The ID of the last contributing
  // Gaussian is known from each pixel from the forward.
  uint32_t contributor = toDo;
  const uint32_t last_contributor = inside ? n_contrib[pix_id] : 0;

  float accum_depth_rec = 0;
  float accum_alpha_rec = 0;
  float accum_normal_rec[3] = {0};

  float dL_dreg;
  float dL_ddepth;
  float dL_daccum;
  float dL_dnormal2D[3] = {0};

  const uint32_t median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
  float dL_dmedian_depth;

  if (inside) {
    dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
    dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
    dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
    dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
    for (int i = 0; i < 3; ++i) {
      dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];
    }
  }

  float last_alpha = 0;
  float last_normal[C] = {0};
  float last_depth = 0;
  const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
  const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
  const float final_A = 1 - T_final;
  float last_dL_dT = 0;

  // Precomputation for efficiency reasons
  const float i_fx = 1.f / projmatrix[0];
  const float i_fy = 1.f / projmatrix[5];

  // Traverse all Gaussians
  for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE) {
    // Load auxiliary data into shared memory, start in the BACK
    // and load them in reverse order
    block.sync();
    const int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y) {
      const int coll_id = point_list[range.y - progress - 1];
      collected_id[block.thread_rank()] = coll_id;
      collected_xy[block.thread_rank()] = points_xy_image[coll_id];
      collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
      collected_Tu[block.thread_rank()] = {
          transMats[12 * coll_id + 0], transMats[12 * coll_id + 1],
          transMats[12 * coll_id + 2], transMats[12 * coll_id + 3]};
      collected_Tv[block.thread_rank()] = {
          transMats[12 * coll_id + 4], transMats[12 * coll_id + 5],
          transMats[12 * coll_id + 6], transMats[12 * coll_id + 7]};
      collected_Tw[block.thread_rank()] = {
          transMats[12 * coll_id + 8], transMats[12 * coll_id + 9],
          transMats[12 * coll_id + 10], transMats[12 * coll_id + 11]};
    }
    block.sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {

      // Keep track of current position in range
      contributor--;
      if (contributor >= last_contributor)
        continue;

      // First compute two homogeneous planes
      const float2 xy = collected_xy[j];
      const float4 Tu = collected_Tu[j];
      const float4 Tv = collected_Tv[j];
      const float4 Tw = collected_Tw[j];

      float2 sph = {i_fx * (pixf.x - projmatrix[8]),
                    i_fy * (pixf.y - projmatrix[9])};
      float3 xyz = sph_to_xyz(sph);
      const float pi_az_inorm = rsqrtf(xyz.y * xyz.y + xyz.x * xyz.x);
      float3 pi_az = {xyz.y * pi_az_inorm, -xyz.x * pi_az_inorm, 0.0f};
      float3 pi_el = cross(pi_az, xyz);
      float3 k = make_float3(dotf4x3(Tu, pi_az), dotf4x3(Tv, pi_az),
                             dotf4x3(Tw, pi_az));
      float3 l = make_float3(dotf4x3(Tu, pi_el), dotf4x3(Tv, pi_el),
                             dotf4x3(Tw, pi_el));
      float3 p = cross(k, l);
      if (p.z == 0.0f)
        continue;
      float2 s = {p.x / p.z, p.y / p.z};

      float rho3d = (s.x * s.x + s.y * s.y);
      float2 d = {xy.x - pixf.x, xy.y - pixf.y};
      float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
      // compute intersection and depth
      float rho = min(rho3d, rho2d);
      float3 beta = tofloat3((s.x * Tu + s.y * Tv) + Tw);
      float valid_tester = dot3f(beta, xyz);
      if (valid_tester < 0.0f)
        continue;
      float depth3d = normf3(beta);
      float depth2d = normf3(tofloat3(Tw));
      float depth = (rho3d <= rho2d) ? depth3d : depth2d;
      if (depth < near_n)
        continue;
      float4 nor_o = collected_normal_opacity[j];
      float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
      float opa = nor_o.w;

      // accumulations
      float power = -0.5f * rho;
      if (power > 0.0f)
        continue;

      const float G = exp(power);
      const float alpha = min(0.99f, opa * G);
      if (alpha < 1.0f / 255.0f)
        continue;
      T = T / (1.f - alpha);
      const float dchannel_dcolor = alpha * T;
      // Propagate gradients to per-Gaussian colors and keep
      // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
      // pair)
      float dL_dalpha = 0.0f;
      const int global_id = collected_id[j];

      // Compute depth distortion gradients
      float dL_dweight = 0;
      float dLd_depth = 0;
      // Pimp the distortion gradient by directly accounting for
      // the depth rather than using the normalized depth with far_n and near_n.
      // const float m_d = far_n / (far_n - near_n) * (1 - near_n / depth);
      // const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * depth *
      // depth);
      const float m_d = depth;
      const float dmd_dd = 1.0f;

      dL_dweight +=
          (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
      dL_dalpha += dL_dweight - last_dL_dT;
      // Propagate current weight W_{i} to the next weight W_{i-1}
      last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
      const float dL_dmd =
          2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
      dLd_depth += dmd_dd * dL_dmd;

      // Integrate median depth
      if (contributor == median_contributor - 1) {
        dLd_depth += dL_dmedian_depth;
      }

      // Compute dL_ddepth and dL_d/dalpha
      dLd_depth += dchannel_dcolor * dL_ddepth;
      accum_depth_rec =
          last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
      last_depth = depth;
      dL_dalpha += (depth - accum_depth_rec) * dL_ddepth;

      // Compute dL_dnormal and dL_normal/d_alpha
      for (int ch = 0; ch < 3; ch++) {
        accum_normal_rec[ch] = last_alpha * last_normal[ch] +
                               (1.f - last_alpha) * accum_normal_rec[ch];
        last_normal[ch] = normal[ch];
        dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
        atomicAdd((&dL_dnormal3D[global_id * 3 + ch]),
                  alpha * T * dL_dnormal2D[ch]);
      }
      // Compute gradients to per-Gaussian alphas
      accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
      dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

      dL_dalpha *= T;
      last_alpha = alpha;
      // Account for fact that alpha also influences how much of
      // the background color is added if nothing left to blend
      float bg_dot_dpixel = 0;

      dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
      atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

      if (rho3d <= rho2d) {
        float beta_dot_Tu = dotf4x3(Tu, beta);
        float beta_dot_Tv = dotf4x3(Tv, beta);
        float2 dL_ds = {-alpha * dL_dalpha * s.x +
                            beta_dot_Tu * alpha * T * dLd_depth / depth,
                        -alpha * dL_dalpha * s.y +
                            beta_dot_Tv * alpha * T * dLd_depth / depth};

        float3 dL_dp = {dL_ds.x / p.z, dL_ds.y / p.z,
                        -(dL_ds.x * p.x + dL_ds.y * p.y) / (p.z * p.z)};
        float3 k1 = cross(dL_dp, k);
        float3 k2 = cross(dL_dp, l);

        const float3 dL_dTu =
            k1.x * pi_el - k2.x * pi_az + (dLd_depth * s.x / depth) * beta;
        const float3 dL_dTv =
            k1.y * pi_el - k2.y * pi_az + (dLd_depth * s.y / depth) * beta;
        const float3 dL_dTw =
            k1.z * pi_el - k2.z * pi_az + (dLd_depth / depth) * beta;

        // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
        atomicAdd(&dL_dtransMat[global_id * 12 + 0], dL_dTu.x);
        atomicAdd(&dL_dtransMat[global_id * 12 + 1], dL_dTu.y);
        atomicAdd(&dL_dtransMat[global_id * 12 + 2], dL_dTu.z);
        // atomicAdd(&dL_dtransMat[global_id * 12 + 3], 0.0f);
        atomicAdd(&dL_dtransMat[global_id * 12 + 4], dL_dTv.x);
        atomicAdd(&dL_dtransMat[global_id * 12 + 5], dL_dTv.y);
        atomicAdd(&dL_dtransMat[global_id * 12 + 6], dL_dTv.z);
        // atomicAdd(&dL_dtransMat[global_id * 12 + 7], 0.0f);
        atomicAdd(&dL_dtransMat[global_id * 12 + 8], dL_dTw.x);
        atomicAdd(&dL_dtransMat[global_id * 12 + 9], dL_dTw.y);
        atomicAdd(&dL_dtransMat[global_id * 12 + 10], dL_dTw.z);
        // atomicAdd(&dL_dtransMat[global_id * 12 + 11], 0.0f);
      } else {
        float2 dL_dd = {-2 * alpha * dL_dalpha * FilterInvSquare * d.x,
                        -2 * alpha * dL_dalpha * FilterInvSquare * d.y};

        const float _ss = Tw.x * Tw.x + Tw.y * Tw.y;
        const float _sn = sqrtf(_ss);

        const float focal_x = projmatrix[0];
        const float focal_y = projmatrix[5];
        float3 dL_dTw = {
            focal_x * dL_dd.x * (-Tw.y / _ss) +
                focal_y * dL_dd.y * (-Tw.x * Tw.z / (_sn * depth2d)),
            focal_x * dL_dd.x * (Tw.x / _ss) +
                focal_y * dL_dd.y * (-Tw.y * Tw.z / (_sn * depth2d)),
            focal_y * dL_dd.y * _sn / depth2d};

        dL_dTw = dL_dTw + (dLd_depth / depth2d) * tofloat3(Tw);
        atomicAdd(&dL_dtransMat[global_id * 12 + 8], dL_dTw.x);
        atomicAdd(&dL_dtransMat[global_id * 12 + 9], dL_dTw.y);
        atomicAdd(&dL_dtransMat[global_id * 12 + 10], dL_dTw.z);
      }
    }
  }
}

__device__ void
compute_transmat_aabb(int idx, const float *Ts_precomp, const float3 *p_origs,
                      const glm::vec2 *scales, const glm::vec4 *rots,
                      const float *projmatrix, const float *viewmatrix,
                      const int W, const int H, const float3 *dL_dnormals,
                      float3 *dL_dmean2Ds, float *dL_dTs, glm::vec3 *dL_dmeans,
                      glm::vec2 *dL_dscales, glm::vec4 *dL_drots) {
  glm::mat3 R;
  glm::mat3 S;
  glm::vec4 rot;
  glm::vec2 scale;

  // Get transformation matrix of the Gaussian
  // p_orig = p_origs[idx];
  rot = rots[idx];
  scale = scales[idx];
  R = quat_to_rotmat(rot);
  S = scale_to_mat(scale, 1.0f);

  glm::mat3 L = R * S;
  glm::mat4 world2sensor =
      glm::mat4(viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
                viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
                viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
                viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]);

  glm::mat3 L_view = world2sensor * glm::mat4(L);
  const float3 dL_dTu = {dL_dTs[idx * 12 + 0], dL_dTs[idx * 12 + 1],
                         dL_dTs[idx * 12 + 2]};
  const float3 dL_dTv = {dL_dTs[idx * 12 + 4], dL_dTs[idx * 12 + 5],
                         dL_dTs[idx * 12 + 6]};
  const float3 dL_dTw = {dL_dTs[idx * 12 + 8], dL_dTs[idx * 12 + 9],
                         dL_dTs[idx * 12 + 10]};
  const float3 dL_dnormal = dL_dnormals[idx];
  glm::mat3 dL_dR =
      glm::mat3(scale.x * dL_dTu.x, scale.x * dL_dTu.y, scale.x * dL_dTu.z,
                scale.y * dL_dTv.x, scale.y * dL_dTv.y, scale.y * dL_dTv.z,
                dL_dnormal.x, dL_dnormal.y, dL_dnormal.z);

  dL_dR = glm::transpose(glm::mat3(world2sensor)) * dL_dR;

  const auto dL_dscalex = glm::dot(glm::vec3(dL_dTu.x, dL_dTu.y, dL_dTu.z),
                                   glm::mat3(world2sensor) * R[0]);
  const auto dL_dscaley = glm::dot(glm::vec3(dL_dTv.x, dL_dTv.y, dL_dTv.z),
                                   glm::mat3(world2sensor) * R[1]);

  dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
  dL_dscales[idx] = glm::vec2(dL_dscalex, dL_dscaley);
  dL_dmeans[idx] =
      glm::vec3(dL_dTw.x, dL_dTw.y, dL_dTw.z) * glm::mat3(world2sensor);

  // As described, we do not need to propagate gradients to means2D
  // since adaptive control is explicitly handled via error maps.
  // Even though we did not include it into the paper, we leave our
  // implementation here :)

  // Project dL_dmeans3Ds to dL_dmeans2Ds using the jacobian of the projection
  // matrix
  // glm::vec3 Tw = glm::vec3(Ts_precomp[12 * idx + 8],
  // Ts_precomp[12 * idx + 9], Ts_precomp[12 * idx + 10]); const float _ss =
  // Tw.x * Tw.x + Tw.y * Tw.y; const float _sn = sqrtf(_ss); const float
  // depth2d = glm::dot(Tw, Tw); glm::mat3x2 J_proj = glm::mat3x2(
  //     -Tw.y / _ss, -Tw.x * Tw.z / (_sn * depth2d),
  //     Tw.x / _ss, -Tw.y * Tw.z / (_sn * depth2d),
  //     0.0f, _sn / depth2d);
  // // glm::vec2 dL_dp2 = J_proj * dL_dmeans[idx];
  // // glm::vec2 dL_dp2 = J_proj * R * dL_dmeans[idx];
  // // glm::vec2 dL_dp2 = J_proj * glm::vec3(dL_dTw.x, dL_dTw.y, dL_dTw.z);
  // glm::vec2 dL_dp2 = J_proj * glm::vec3(dL_dTu.x, dL_dTu.y, dL_dTu.z) +
  // J_proj * glm::vec3(dL_dTv.x, dL_dTv.y, dL_dTv.z) + J_proj *
  // glm::vec3(dL_dTw.x, dL_dTw.y, dL_dTw.z); dL_dmean2Ds[idx].x = dL_dp2.x;
  // dL_dmean2Ds[idx].y = dL_dp2.y;
}

template <int C>
__global__ void preprocessCUDA(
    int P, const float3 *means3D, const float *transMats, const int *radii,
    const bool *clamped, const glm::vec2 *scales, const glm::vec4 *rotations,
    const float scale_modifier, const float *viewmatrix,
    const float *projmatrix, const int W, const int H,
    // grad input
    float *dL_dtransMats, const float *dL_dnormal3Ds, float3 *dL_dmean2Ds,
    glm::vec3 *dL_dmean3Ds, glm::vec2 *dL_dscales, glm::vec4 *dL_drots) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= static_cast<unsigned long long>(P) || !(radii[idx] > 0))
    return;

  // const float *Ts_precomp = (scales) ? nullptr : transMats;
  const float *Ts_precomp = transMats;

  compute_transmat_aabb(idx, Ts_precomp, means3D, scales, rotations, projmatrix,
                        viewmatrix, W, H, (float3 *)dL_dnormal3Ds, dL_dmean2Ds,
                        dL_dtransMats, dL_dmean3Ds, dL_dscales, dL_drots);
}
void BACKWARD::preprocess(int P, const float3 *means3D, const int *radii,
                          const bool *clamped, const glm::vec2 *scales,
                          const glm::vec4 *rotations,
                          const float scale_modifier, const float *transMats,
                          const float *viewmatrix, const float *projmatrix,
                          const int W, const int H, float3 *dL_dmean2Ds,
                          const float *dL_dnormal3Ds, float *dL_dtransMats,
                          glm::vec3 *dL_dmean3Ds, glm::vec2 *dL_dscales,
                          glm::vec4 *dL_drots) {
  preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
      P, (float3 *)means3D, transMats, radii, clamped, (glm::vec2 *)scales,
      (glm::vec4 *)rotations, scale_modifier, viewmatrix, projmatrix, W, H,
      dL_dtransMats, dL_dnormal3Ds, dL_dmean2Ds, dL_dmean3Ds, dL_dscales,
      dL_drots);
}

void BACKWARD::render(const dim3 grid, const dim3 block, const uint2 *ranges,
                      const uint32_t *point_list, int W, int H,
                      const float *projmatrix, const float2 *means2D,
                      const float4 *normal_opacity, const float *transMats,
                      const float *depths, const float *final_Ts,
                      const uint32_t *n_contrib, const float *dL_depths,
                      float *dL_dtransMat, float3 *dL_dmean2D,
                      float *dL_dnormal3D, float *dL_dopacity) {
  renderCUDA<NUM_CHANNELS><<<grid, block>>>(
      ranges, point_list, W, H, projmatrix, means2D, normal_opacity, transMats,
      depths, final_Ts, n_contrib, dL_depths, dL_dtransMat, dL_dmean2D,
      dL_dnormal3D, dL_dopacity);
}
