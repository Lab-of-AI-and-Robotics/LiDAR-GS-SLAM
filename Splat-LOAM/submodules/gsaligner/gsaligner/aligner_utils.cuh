#pragma once
#include "registration/photometric.cuh"
#include <torch/extension.h>

void init_photo_image_from_buffer(const torch::Tensor buf, photo::Image& img, const int rows, const int cols);