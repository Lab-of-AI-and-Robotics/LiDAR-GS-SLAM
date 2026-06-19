#include "aligner_utils.cuh"

__global__ void init_photo_image_from_buffer_kernel(const float* buf, photo::Image* img, const int rows, const int cols) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= rows || col >= cols)
    return;

  const float dval = buf[row * cols + col];
  if (dval > 1e-5)
    img->at<1>(row, col).setMasked(false);
  img->at<1>(row, col).setDepth(dval);

  // printf("row: %d, col: %d, dval: %f, masked=%d\t", row, col, dval, img->at<1>(row, col).masked());
}

void init_photo_image_from_buffer_cpu(const torch::Tensor buf, photo::Image& img, const int rows, const int cols) {
  torch::Tensor buf_cpu = buf.to(torch::kCPU);
  auto buf_acc          = buf_cpu.accessor<float, 3>();
  // const float* buf_ptr = buf.contiguous().data<float>();
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      // const float dval = buf_ptr[row * cols + col];
      const float dval = buf_acc[0][row][col];
      if (dval > 1e-5)
        img.at(row, col).setMasked(false);
      img.at(row, col).setDepth(dval);
    }
  }
}

void init_photo_image_from_buffer(const torch::Tensor buf, photo::Image& img, const int rows, const int cols) {
  init_photo_image_from_buffer_kernel<<<dim3((cols + 16 - 1) / 16, (rows + 16 - 1) / 16), dim3(16, 16)>>>(
    buf.contiguous().data<float>(), img.deviceInstance(), rows, cols);
  cudaDeviceSynchronize();
  // init_photo_image_from_buffer_cpu(buf, img, rows, cols);
  // img.toDevice();
}
