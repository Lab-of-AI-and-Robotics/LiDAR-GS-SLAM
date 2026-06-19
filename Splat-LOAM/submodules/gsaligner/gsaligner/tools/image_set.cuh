
#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

#include "dual_matrix.cuh"
#include "utils.h"
#include "utils_photo.cuh"

namespace photo {

using Matrix5_6f = Eigen::Matrix<float, 5, 6>;
using Matrix5_2f = Eigen::Matrix<float, 5, 2>;
using Vector5f = Eigen::Matrix<float, 5, 1>;

enum ChannelType { Intensity = 0x0, Depth = 0x1, Normal = 0x2 };
enum FilterPolicy { Ignore = 0, Suppress = 1, Clamp = 2 };
enum PointStatusFlag {
  Good = 0x00,
  Outside = 0x1,
  DepthOutOfRange = 0x2,
  Masked = 0x3,
  Occluded = 0x4,
  DepthError = 0x5,
  Invalid = 0x7,
};

template <typename T>
struct Point_ {
  // clang-format off
    Eigen::Matrix<T, 3, 1> coordinates = Eigen::Matrix<T, 3, 1>::Zero();    
    Eigen::Matrix<T, 3, 1> normal = Eigen::Matrix<T, 3, 1>::Zero() ;   
    T intensity = T(0);
    PointStatusFlag status = PointStatusFlag::Invalid;
  // clang-format on
};

using Pointf = Point_<float>;
using MatrixCloud = DualMatrix_<Pointf>;  // matrix vector cloud and matrix
                                          // cloud can be taken from here
using VectorCloud = std::vector<Pointf>;

// ImageEntry: represents an entry of a Phometric Image,
// in terms of both point and derivatives
struct ImageEntry {
  Matrix5_2f derivatives;  // col and row derivatives
  Vector5f value;          // [i, d, nx, ny, nz]: intensity, depth, normal
#ifdef _MD_ENABLE_SUPERRES_
  float r = 0, c = 0;  // subpixel coordinates
#endif
  bool msked = true;
  ImageEntry() {
    value.setZero();
    derivatives.setZero();
  }
  // clang-format off
    __host__ __device__ inline float intensity() const { return value(0); }
    __host__ __device__ inline void setIntensity(float intensity) { value(0) = intensity; }
    __host__ __device__ inline float depth() const { return value(1); }
    __host__ __device__ inline void setDepth(float depth) { value(1) = depth; }
    __host__ __device__ inline Eigen::Vector3f normal() const { return value.block<3, 1>(2, 0); }
    __host__ __device__ inline void setNormal(const Eigen::Vector3f& n) { value.block<3, 1>(2, 0) = n; }
    __host__ __device__ inline bool masked() const { return msked; }
    __host__ __device__ inline void setMasked(bool masked) { msked = masked; }
  // clang-format on
};

using Image = DualMatrix_<ImageEntry>;

__device__ __forceinline__ bool getSubPixel(
    Vector5f& value, Matrix5_2f& derivative, const Image* mat,
    const Eigen::Vector2f& image_point) {
  float c = image_point.x();
  float r = image_point.y();
  int r0 = (int)r;
  int c0 = (int)c;
  if (!mat->inside(r0, c0)) return false;

  int r1 = r0 + 1;
  int c1 = c0 + 1;
  if (!mat->inside(r1, c1)) return false;

  const ImageEntry p00 = mat->at<1>(r0, c0);
  const ImageEntry p01 = mat->at<1>(r0, c1);
  const ImageEntry p10 = mat->at<1>(r1, c0);
  const ImageEntry p11 = mat->at<1>(r1, c1);
  if (p00.masked() || p01.masked() || p10.masked() || p11.masked())
    return false;

  const float dr = r - (float)r0;
  const float dc = c - (float)c0;
  const float dr1 = 1.f - dr;
  const float dc1 = 1.f - dc;

  value = (p00.value * dc1 + p01.value * dc) * dr1 +
          (p10.value * dc1 + p11.value * dc) * dr;

  derivative = (p00.derivatives * dc1 + p01.derivatives * dc) * dr1 +
               (p10.derivatives * dc1 + p11.derivatives * dc) * dr;

  return true;
}

// containing some "static" image attributes
struct ImageAttributes {
  // parameters used to compute the derivatives and the mask
  float thresholds[3] = {10.f, 0.5f, 0.5f};
  FilterPolicy policies[3] = {Ignore, Suppress, Clamp};
  float min_depth = 0.f;  // minimum depth of the cloud
  float max_depth = 0.f;  // max depth of the cloud
  int mask_grow_radius = 0;
};
using ImageAttributesPtr = std::shared_ptr<ImageAttributes>;

// ImageSet: contains image data (intensity, depth, normals), i.e.
//  - the image as a Image
//  - a mask of valid points
//  - camera matrix at this level of pyramid
//  - image size
//  - the corresponding cloud
// the ImageProcessor takes care of initialize Levels
class ImageSet {
 public:
  ~ImageSet();

  //! @brief PyramidLevel c'tor
  ImageSet(const size_t rows = 0, const size_t cols = 0);

  //! @brief resizes image and mask
  void resize(const size_t rows, const size_t cols);

  // clang-format off
    const size_t rows() const { return matrix_.rows(); }
    const size_t cols() const { return matrix_.cols(); }
    const double& getTimestamp() const { return timestamp_; }
  // clang-format on

  //! @brief generates a cloud based on the pyramid
  void toCloud(
      MatrixCloud& target) const;  // puts a cloud rendered from self in target
  __host__ void toCloudDevice(
      MatrixCloud* target) const;  // does the same but in device

  //! @brief generates the level based on the cloud
  //! the matrix should be resized before calling the method
  //! uses the embedded parameters
  void fromCloud(
      const MatrixCloud& src_cloud);  // fills own values from src_cloud

  //! @brief produces a 3x3  tiled image of the pyramid for debug
  void showImageSet(int key);

  //! @brief: these are to get a cue stripped from the pyramid level
  void getIntensity(Matrixf& intensity) const;
  void getDepth(Matrixf& depth) const;
  void getNormals(Matrixf3& normals) const;

  // protected:
  void growMask();
  void updateDerivatives();

  void growMaskDevice();
  void updateDerivativesDevice();
  // void updateDerivativesDevice(Image& matrix_, const float
  // threshold_intensity,
  //                              const float threshold_depth,
  //                              const float threshold_normal,
  //                              const FilterPolicy policy_intensity,
  //                              const FilterPolicy policy_depth,
  //                              const FilterPolicy policy_normal);

  SensorPtr sensor_;
  Image matrix_;  // TODO change name - image with all stuff;
  ImageAttributesPtr attributes_;

  double timestamp_ = 0.0;  // timestamp of images
};

}  // namespace photo
