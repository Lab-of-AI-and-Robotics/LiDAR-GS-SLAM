#pragma once
#include <tools/lie_algebra.h>
#include <tools/linear_system_entry.h>

#include <cfloat>
#include <tools/dual_matrix.cuh>
#include <tools/image_set.cuh>

namespace photo {

using Matrix5_6f = Eigen::Matrix<float, 5, 6>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6f = Eigen::Matrix<float, 3, 6>;
using Matrix2_3f = Eigen::Matrix<float, 2, 3>;
using Matrix1_3f = Eigen::Matrix<float, 1, 3>;

__device__ __forceinline__ void computeAtxA(Matrix6d& dest,
                                            const Matrix5_6f& src,
                                            const float& lambda) {
  for (int c = 0; c < 6; ++c) {
    for (int r = 0; r <= c; ++r) {
      dest(r, c) += static_cast<double>(src.col(r).dot(src.col(c)) * lambda);
    }
  }
}

__host__ inline void copyLowerTriangleUp(Matrix6d& A) {
  for (int c = 0; c < 6; ++c) {
    for (int r = 0; r < c; ++r) {
      A(c, r) = A(r, c);
    }
  }
}

__host__ inline Matrix6d vectorToMat6DUpperTriangular(const Vector21d& vec) {
  Matrix6d mat = Matrix6d::Zero();
  int k = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = i; j < 6; ++j) {
      mat(i, j) = vec(k++);
    }
  }
  return mat;
}

__device__ __forceinline__ Vector21d
mat6DUpperTriangularToVector(const Matrix6d& mat) {
  Vector21d vec;
  int k = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = i; j < 6; ++j) {
      vec(k++) = mat(i, j);
    }
  }
  return vec;
}

__device__ __forceinline__ unsigned long long packIntFloat(int a, float b) {
  return (((unsigned long long)(*(reinterpret_cast<unsigned*>(&b)))) << 32) +
         *(reinterpret_cast<unsigned*>(&a));
}

__device__ __forceinline__ void unpackIntFloat(int& a, float& b,
                                               unsigned long long val) {
  unsigned ma = (unsigned)(val & 0x0FFFFFFFFULL);
  a = *(reinterpret_cast<int*>(&ma));
  unsigned mb = (unsigned)(val >> 32);
  b = *(reinterpret_cast<float*>(&mb));
}

// clang-format off
  struct __align__(16) WorkspaceEntry {
    __host__ __device__ WorkspaceEntry() { prediction[1] = FLT_MAX; }
    Eigen::Matrix<float, 5, 1> prediction = Eigen::Matrix<float, 5, 1>::Zero();
    Eigen::Matrix<float, 5, 1> error = Eigen::Matrix<float, 5, 1>::Zero();
    Eigen::Vector3f point = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    Eigen::Vector3f transformedPoint = Eigen::Vector3f::Zero();
    Eigen::Vector3f cameraPoint = Eigen::Vector3f::Zero();
    Eigen::Vector2f imagePoint = Eigen::Vector2f::Zero();
    int index = -1;
    float chi = 0.f;
    unsigned long long depthIdx = ULLONG_MAX;
    PointStatusFlag status = PointStatusFlag::Good;

    __host__ __device__ const float intensity() { return prediction[0]; }
    __host__ __device__ const float depth() { return prediction[1]; }
  };
// clang-format on

using Workspace = DualMatrix_<WorkspaceEntry>;

// usage with float
using MatrixCloud = DualMatrix_<Pointf>;

class __align__(16) PhotometricFac {
 public:
  ~PhotometricFac() {
    if (workspace_) delete workspace_;
    if (entry_array_) cudaFree(entry_array_);
  }

  // clang-format off
    PhotometricFac();

    void setFixedData(ImageSet* refData);
    void setMovingData(const ImageSet& currData);
    void computeProjections();
    void setTransform(const Eigen::Isometry3f& transform);

    inline float omegaDepth() const { return omegaDepth_; }
    inline float omegaIntensity() const { return omegaIntensity_; }
    inline float omegaNormals() const { return omegaNormals_; }
    inline float depthRejectionThreshold() const { return depthRejectionThreshold_; }
    inline float kernelChiThreshold() const { return kernelChiThreshold_; }
    inline void setOmegaDepth(float v) { omegaDepth_ = v; }
    inline void setOmegaIntensity(float v) { omegaIntensity_ = v; }
    inline void setOmegaNormals(float v) { omegaNormals_ = v; }
    inline void setDepthRejectionThreshold(float v) { depthRejectionThreshold_ = v; }
    inline void setKernelChiThreshold(float v) { kernelChiThreshold_ = v; }

    inline const Eigen::Isometry3f& transform() const{ return X_; };
    std::pair<float, float> compute(const size_t max_iterations, const bool print_stats);
    std::pair<float, float> linearize();
  // clang-format on

  //  protected:
 public:
  MatrixCloud currData_;
  Eigen::Isometry3f X_;
  Eigen::Isometry3f sensorOffsetInverse_;
  Eigen::Isometry3f SX_;
  Eigen::Matrix3f neg2rotSX_;
  Eigen::Matrix3f sensorOffsetRotationInverse_;
  Eigen::Matrix3f cameraMatrix_;
  CameraType camType_;
  ImageSet* referenceData_ = nullptr;         // fixed data, managed outside
  Workspace* workspace_ = nullptr;            // allocated inside
  LinearSystemEntry* entry_array_ = nullptr;  // allocated inside
  float omegaIntensity_ = 0.f;
  float omegaDepth_ = 0.f;
  float omegaNormals_ = 0.f;
  float depthRejectionThreshold_ = 0.f;
  float kernelChiThreshold_ = 0.f;
  float minDepth_;
  float maxDepth_;
  int rows_;
  int cols_;
  int entry_array_size_;
};

using PhotometricFacPtr = std::shared_ptr<PhotometricFac>;
}  // namespace photo
