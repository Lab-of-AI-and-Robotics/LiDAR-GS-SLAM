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
#include "forward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(const float3 &p_orig, const glm::vec2 scale,
                                 float mod, const glm::vec4 rot, const float *,
                                 const float *viewmatrix, const int, const int,
                                 glm::mat3x4 &T, float3 &normal) {
  glm::mat4 world2sensor =
      glm::mat4(viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
                viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
                viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
                viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]);

  glm::mat3 R = quat_to_rotmat(rot);
  glm::mat3 S = scale_to_mat(scale, mod);
  glm::mat3 L = R * S;

  glm::mat3 L_view = world2sensor * glm::mat4(L);
  glm::vec3 p_view =
      world2sensor * glm::vec4(glm::vec3(p_orig.x, p_orig.y, p_orig.z), 1);

  glm::mat3x4 splat2world =
      glm::mat3x4(glm::vec4(L_view[0], 0.0), glm::vec4(L_view[1], 0.0),
                  glm::vec4(p_view, 1.0));

  T = splat2world;
  normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
}

// Section 3.2.3. Bounding Box Computation
__device__ bool compute_aabb(glm::mat3 T, const float *projmatrix, float cutoff,
                             const int W, const int _, float2 &point_image,
                             float2 &extent) {
  glm::vec3 p_center(0.0, 0.0, 1.0);
  glm::vec3 p0(cutoff, cutoff, 1.0);
  glm::vec3 p1(cutoff, -cutoff, 1.0);
  glm::vec3 p2(-cutoff, -cutoff, 1.0);
  glm::vec3 p3(-cutoff, cutoff, 1.0);
  glm::vec3 p_center_sensor = T * glm::vec4(p_center, 1.0);
  glm::vec3 p0_sensor = T * glm::vec4(p0, 1.0);
  glm::vec3 p1_sensor = T * glm::vec4(p1, 1.0);
  glm::vec3 p2_sensor = T * glm::vec4(p2, 1.0);
  glm::vec3 p3_sensor = T * glm::vec4(p3, 1.0);
  glm::vec2 p_center_s = xyz_to_sph(p_center_sensor);
  glm::vec2 p0_s = xyz_to_sph(p0_sensor);
  glm::vec2 p1_s = xyz_to_sph(p1_sensor);
  glm::vec2 p2_s = xyz_to_sph(p2_sensor);
  glm::vec2 p3_s = xyz_to_sph(p3_sensor);
  // Project the points onto the image plane
  point_image = {p_center_s.x * projmatrix[0] + projmatrix[8],
                 p_center_s.y * projmatrix[5] + projmatrix[9]};
  float2 p0_img = {p0_s.x * projmatrix[0] + projmatrix[8],
                   p0_s.y * projmatrix[5] + projmatrix[9]};
  float2 p1_img = {p1_s.x * projmatrix[0] + projmatrix[8],
                   p1_s.y * projmatrix[5] + projmatrix[9]};
  float2 p2_img = {p2_s.x * projmatrix[0] + projmatrix[8],
                   p2_s.y * projmatrix[5] + projmatrix[9]};
  float2 p3_img = {p3_s.x * projmatrix[0] + projmatrix[8],
                   p3_s.y * projmatrix[5] + projmatrix[9]};

  float delta_x = 0.5 * W - point_image.x;

  float2 point_image_normalized = {point_image.x + delta_x, point_image.y};
  float2 p0_img_normalized = {fmodf(fmodf(p0_img.x + delta_x, W) + W, W),
                              p0_img.y};
  float2 p1_img_normalized = {fmodf(fmodf(p1_img.x + delta_x, W) + W, W),
                              p1_img.y};
  float2 p2_img_normalized = {fmodf(fmodf(p2_img.x + delta_x, W) + W, W),
                              p2_img.y};
  float2 p3_img_normalized = {fmodf(fmodf(p3_img.x + delta_x, W) + W, W),
                              p3_img.y};

  float max_extent =
      0.5 * max(max(normf2(point_image_normalized - p0_img_normalized),
                    normf2(point_image_normalized - p1_img_normalized)),
                max(normf2(point_image_normalized - p2_img_normalized),
                    normf2(point_image_normalized - p3_img_normalized)));

  extent = make_float2(max_extent, max_extent);

  return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C>
__global__ void
preprocessCUDA(int P, const float *orig_points, const glm::vec2 *scales,
               const float scale_modifier, const glm::vec4 *rotations,
               const float *opacities, bool *, const float *transMat_precomp,
               const float *viewmatrix, const float *projmatrix, const int W,
               int H, int *radii, float2 *points_xy_image, float *depths,
               float *transMats, float4 *normal_opacity, const dim3 grid,
               uint32_t *tiles_touched, bool prefiltered) {

  auto idx = cg::this_grid().thread_rank();
  if (idx >= static_cast<unsigned long long>(P))
    return;

  // Initializeradius and touched tiles to 0. If this isn't changed,
  // this Gaussian will not be processed further.
  radii[idx] = 0;
  tiles_touched[idx] = 0;

  float3 p_view;
  if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered,
                  p_view))
    return;

  // Compute transformation matrix
  glm::mat3x4 T;
  float3 normal;
  if (transMat_precomp == nullptr) {
    compute_transmat(((float3 *)orig_points)[idx], scales[idx], scale_modifier,
                     rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
    float4 *T_ptr = (float4 *)transMats;
    T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2], T[0][3]};
    T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2], T[1][3]};
    T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2], T[2][3]};
  } else {
    glm::vec4 *T_ptr = (glm::vec4 *)transMat_precomp;
    T = glm::mat3x4(T_ptr[idx * 3 + 0], T_ptr[idx * 3 + 1], T_ptr[idx * 3 + 2]);
    normal = make_float3(0.0, 0.0, 1.0);
  }

  float cutoff = 3.0f;

// Hide back-facing splats
#ifdef BACKFACE_CULL
  if (-sumf3(p_view * normal) <= 0.0f)
    return;
#endif

  // Compute center and radius
  float2 point_image;
  float radius;
  {
    float2 extent;
    bool ok = compute_aabb(T, projmatrix, cutoff, W, H, point_image, extent);
    if (!ok)
      return;
    radius = ceil(max(extent.x, extent.y)) * 2;
  }

  // Grid filtering
  int2 rect_min, rect_max;
  getRectSpherical(point_image, radius, rect_min, rect_max, grid);
  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
    return;

  depths[idx] = normf3(p_view);
  radii[idx] = (int)radius;
  points_xy_image[idx] = point_image;
  normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderCUDA(const uint2 *__restrict__ ranges,
               const uint32_t *__restrict__ point_list, int W, int H,
               const float *__restrict__ projmatrix,
               const float2 *__restrict__ points_xy_image,
               const float *__restrict__ transMats, const float *__restrict__,
               const float4 *__restrict__ normal_opacity,
               float *__restrict__ final_T, uint32_t *__restrict__ n_contrib,
               float *__restrict__ out_others) {
  // Identify current tile and associated min/max pixel range
  auto block = cg::this_thread_block();
  uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  uint2 pix_min = {block.group_index().x * BLOCK_X,
                   block.group_index().y * BLOCK_Y};
  uint2 pix = {pix_min.x + block.thread_index().x,
               pix_min.y + block.thread_index().y};
  uint32_t pix_id = W * pix.y + pix.x;
  float2 pixf = {(float)pix.x, (float)pix.y};

  // Check if this thread is associated with a valid pixel or outside
  bool inside = pix.x < static_cast<float>(W) && pix.y < static_cast<float>(H);
  // Done threads can help with fetching, but don't rasterize
  bool done = !inside;

  // Load start/end range of IDs to process in bit sorted list.
  uint2 range =
      ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int toDo = range.y - range.x;

  // Allocate storage for batches of collectively fetched data
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_normal_opacity[BLOCK_SIZE];
  __shared__ float4 collected_Tu[BLOCK_SIZE];
  __shared__ float4 collected_Tv[BLOCK_SIZE];
  __shared__ float4 collected_Tw[BLOCK_SIZE];

  // Initialize helper variables
  float T = 1.0f;
  uint32_t contributor = 0;
  uint32_t last_contributor = 0;

  float D = 0;      // Depth Channel
  float N[3] = {0}; // Normal Channel
  float M1 = {0};
  float M2 = {0};
  float distortion = {0};
  float median_depth = {0};
  float median_contributor = {-1};

  // Precomputation for efficiency reasons

  const float i_fx = 1.f / projmatrix[0];
  const float i_fy = 1.f / projmatrix[5];

  // Iterate over batches until all done or range is complete
  for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE) {
    // End if entire block votes that it is done rasterizing
    int num_done = __syncthreads_count(done);
    if (num_done == BLOCK_SIZE)
      break;

    // Collectively fetch per-Gaussian data from global to shared
    int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y) {
      unsigned int coll_id = point_list[range.x + progress];
      collected_xy[block.thread_rank()] = points_xy_image[coll_id];
      collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
      const float4 *transMats4 = reinterpret_cast<const float4 *>(transMats);
      collected_Tu[block.thread_rank()] = transMats4[coll_id * 3 + 0];
      collected_Tv[block.thread_rank()] = transMats4[coll_id * 3 + 1];
      collected_Tw[block.thread_rank()] = transMats4[coll_id * 3 + 2];
    }
    block.sync();

    // Iterate over current batch
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // Keep track of current position in range
      contributor++;

      // First compute two homogeneous planes, See Eq. (8)
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
      // Culling negative or non intersecting splats
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

      float power = -0.5f * rho;
      if (power > 0.0f)
        continue;

      float alpha = min(0.99f, opa * exp(power));
      if (alpha < 1.0 / 255.0f)
        continue;
      float test_T = T * (1 - alpha);
      if (test_T < 0.0001f) {
        done = true;
        continue;
      }
      float w = alpha * T;
      // Compute distortion depth
      const float A = 1 - T;
      // float m = far_n / (far_n - near_n) * (1 - near_n / depth);
      float m = depth;
      distortion += (m * m * A + M2 - 2 * m * M1) * w;
      M1 += m * w;
      M2 += m * m * w;

      D += depth * w;

      // Median Depth
      if (T > 0.5) {
        median_depth = depth;
        median_contributor = contributor;
      }

      for (int ch = 0; ch < 3; ++ch)
        N[ch] += normal[ch] * w;

      T = test_T;

      // Keep track of last range entry to update this pixel
      last_contributor = contributor;
    }
  }

  // All threads that treat valid pixel write out their final
  // rendering data to the frame and auxiliary buffers
  if (inside) {
    final_T[pix_id] = T;
    n_contrib[pix_id] = last_contributor;
    out_others[pix_id + DEPTH_OFFSET * H * W] = D;
    out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
    for (int ch = 0; ch < 3; ++ch)
      out_others[pix_id + (NORMAL_OFFSET + ch) * H * W] = N[ch];

    // Add depth distortion contributions
    final_T[pix_id + H * W] = M1;
    final_T[pix_id + 2 * H * W] = M2;
    out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
    n_contrib[pix_id + H * W] = median_contributor;
    out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
  }
}

void FORWARD::render(const dim3 grid, dim3 block, const uint2 *ranges,
                     const uint32_t *point_list, int W, int H,
                     const float *projmatrix, const float2 *means2D,
                     const float *transMats, const float *depths,
                     const float4 *normal_opacity, float *final_T,
                     uint32_t *n_contrib, float *out_others) {
  renderCUDA<NUM_CHANNELS><<<grid, block>>>(
      ranges, point_list, W, H, projmatrix, means2D, transMats, depths,
      normal_opacity, final_T, n_contrib, out_others);
}

void FORWARD::preprocess(int P, const float *means3D, const glm::vec2 *scales,
                         const float scale_modifier, const glm::vec4 *rotations,
                         const float *opacities, bool *clamped,
                         const float *transMat_precomp, const float *viewmatrix,
                         const float *projmatrix, const int W, const int H,
                         int *radii, float2 *means2D, float *depths,
                         float *transMats, float4 *normal_opacity,
                         const dim3 grid, uint32_t *tiles_touched,
                         bool prefiltered) {
  preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
      P, means3D, scales, scale_modifier, rotations, opacities, clamped,
      transMat_precomp, viewmatrix, projmatrix, W, H, radii, means2D, depths,
      transMats, normal_opacity, grid, tiles_touched, prefiltered);
}
