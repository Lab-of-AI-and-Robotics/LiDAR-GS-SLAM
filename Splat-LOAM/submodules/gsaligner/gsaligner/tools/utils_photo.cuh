#pragma once

#include <Eigen/Dense>
#include <iterator>
#include <memory>

namespace lie {

  __device__ __forceinline__ Eigen::Matrix3f skew(const Eigen::Vector3f& v) {
    Eigen::Matrix3f S;
    S << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
    return S;
  }

} // namespace lie

namespace photo {

  enum CameraType { Pinhole = 0, Spherical = 1, Unknown = -1 };

  struct Sensor {
    // clang-format off
    inline const Eigen::Isometry3f& sensorOffset() const { return sOffset; }
    inline const Eigen::Matrix3f& cameraMatrix() const { return camMatrix; }
    inline const CameraType cameraType() const { return camType; }
    inline const int rows() const { return rws; }
    inline const int cols() const { return cls; }
    inline const float depthScale() const { return dScale; }

    inline Eigen::Isometry3f& sensorOffset() { return sOffset; }
    inline Eigen::Matrix3f& cameraMatrix() { return camMatrix; }
    inline CameraType& cameraType() { return camType; }
    inline void setDepthScale(float dScale) { dScale = dScale; }
    inline int& rows() { return rws; }
    inline int& cols() { return cls; }
    // clang-format on

  protected:
    Eigen::Isometry3f sOffset = Eigen::Isometry3f::Identity();
    Eigen::Matrix3f camMatrix = Eigen::Matrix3f::Zero();
    CameraType camType        = CameraType::Pinhole;
    float dScale              = 0.f;
    int rws                   = 0;
    int cls                   = 0;
  };

  using SensorPtr = std::shared_ptr<Sensor>;

  __host__ __device__ inline bool project(Eigen::Vector2f& image_point,
                                          Eigen::Vector3f& camera_point,
                                          float& depth,
                                          const Eigen::Vector3f& point,
                                          const CameraType& camera_type,
                                          const Eigen::Matrix3f& camera_mat,
                                          const float min_depth,
                                          const float max_depth) {
    switch (camera_type) {
      case CameraType::Pinhole:
        depth = point.z();
        if (depth < min_depth || depth > max_depth) {
          return false;
        }
        camera_point = camera_mat * point;
        image_point  = camera_point.head(2) * 1.f / depth;
        break;
      case CameraType::Spherical:
        depth = point.norm();
        if (depth < min_depth || depth > max_depth) {
          return false;
        }
        camera_point.x() = atan2(point.y(), point.x());
        camera_point.y() = atan2(point.z(), point.head<2>().norm());
        camera_point.z() = depth;

        image_point.x() = camera_mat(0, 0) * camera_point.x() + camera_mat(0, 2);
        image_point.y() = camera_mat(1, 1) * camera_point.y() + camera_mat(1, 2);
        break;
        // default:
        //   throw std::runtime_error("utils::project | unknown camera type");
    }
    return true;
  }
} // namespace photo