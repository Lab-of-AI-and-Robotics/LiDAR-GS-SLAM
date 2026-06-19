#pragma once
#include <torch/extension.h>

#include <cstdio>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &means3D, const torch::Tensor &opacity,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &transMat_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const int image_height, const int image_width, const bool prefiltered,
    const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &means3D, const torch::Tensor &radii,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const float scale_modifier, const torch::Tensor &transMat_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const torch::Tensor &dL_dout_others, const torch::Tensor &geomBuffer,
    const int R, const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer, const bool debug);

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix);
