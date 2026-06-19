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

#include "cuda_rasterizer/rasterizer.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <functional>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <tuple>

#define CHECK_INPUT(x)                                                         \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long)N});
    return reinterpret_cast<char *>(t.contiguous().data_ptr());
  };
  return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &means3D, const torch::Tensor &opacity,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &transMat_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const int image_height, const int image_width, const bool prefiltered,
    const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(means3D);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);

  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_others = torch::full({3 + 3 + 1, H, W}, 0.0, float_opts);
  torch::Tensor radii =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    rendered = CudaRasterizer::Rasterizer::forward(
        geomFunc, binningFunc, imgFunc, P, W, H,
        means3D.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(), scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        transMat_precomp.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(), prefiltered,
        out_others.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(), debug);
  }
  return std::make_tuple(rendered, out_others, radii, geomBuffer, binningBuffer,
                         imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &means3D, const torch::Tensor &radii,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &transMat_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const torch::Tensor &dL_dout_others, const torch::Tensor &geomBuffer,
    const int R, const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer, const bool debug) {

  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  const int P = means3D.size(0);
  const int H = dL_dout_others.size(1);
  const int W = dL_dout_others.size(2);

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 12}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if (P != 0) {
    CudaRasterizer::Rasterizer::backward(
        P, R, W, H, means3D.contiguous().data_ptr<float>(),
        scales.data_ptr<float>(), scale_modifier, rotations.data_ptr<float>(),
        transMat_precomp.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
        dL_dout_others.contiguous().data_ptr<float>(),
        dL_dmeans2D.contiguous().data_ptr<float>(),
        dL_dnormal.contiguous().data_ptr<float>(),
        dL_dopacity.contiguous().data_ptr<float>(),
        dL_dmeans3D.contiguous().data_ptr<float>(),
        dL_dtransMat.contiguous().data_ptr<float>(),
        dL_dscales.contiguous().data_ptr<float>(),
        dL_drotations.contiguous().data_ptr<float>(), debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dopacity, dL_dmeans3D, dL_dtransMat,
                         dL_dscales, dL_drotations);
}

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix) {
  const int P = means3D.size(0);

  torch::Tensor present =
      torch::full({P}, false, means3D.options().dtype(at::kBool));

  if (P != 0) {
    CudaRasterizer::Rasterizer::markVisible(
        P, means3D.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        present.contiguous().data_ptr<bool>());
  }

  return present;
}
