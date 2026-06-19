#pragma once
#include "cuda_utils.cuh"
#include <Eigen/Dense>

namespace photo {

  using Vector21d = Eigen::Matrix<double, 21, 1>;
  using Vector6d  = Eigen::Matrix<double, 6, 1>; // Replacing dependency on srrg_geometry

  struct LinearSystemEntry {
    Vector21d upperH;
    Vector6d b;
    size_t is_good;
    size_t is_inlier;
    float chi;

    __host__ __device__ LinearSystemEntry() : upperH(Vector21d::Zero()), b(Vector6d::Zero()), chi(0.f), is_good(0), is_inlier(0) {
    }

    __host__ __device__ LinearSystemEntry(const LinearSystemEntry& entry_) = default;

    __host__ __device__ LinearSystemEntry& operator=(const LinearSystemEntry& other_) = default;

    __host__ __device__ LinearSystemEntry& operator+=(const LinearSystemEntry& other_) {
      upperH += other_.upperH;
      b += other_.b;
      chi += other_.chi;
      is_good += other_.is_good;
      is_inlier += other_.is_inlier;
      return *this;
    }

    __host__ __device__ LinearSystemEntry operator+(const LinearSystemEntry& other_) const {
      LinearSystemEntry result = *this;
      result += other_;
      return result;
    }

    __host__ static LinearSystemEntry Ones() {
      LinearSystemEntry e;
      e.upperH.setOnes();
      e.b.setOnes();
      e.chi = 1.f;
      return e;
    }

    __host__ static LinearSystemEntry Random() {
      LinearSystemEntry e;
      e.upperH.setRandom();
      e.b.setRandom();
      e.chi = static_cast<float>(rand()) / RAND_MAX;
      return e;
    }
  };

} // namespace photo