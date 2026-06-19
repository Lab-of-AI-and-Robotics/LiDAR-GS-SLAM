#include "photometric.cuh"

#include <cassert>
#include <sys/time.h>

#include <tools/dual_matrix.cuh>

namespace photo {

  PhotometricFac::PhotometricFac() {
    rows_     = 0;
    cols_     = 0;
    minDepth_ = 0.f;
    maxDepth_ = 0.f;
    // TODO propagate this shit
    omegaIntensity_          = 1.f;
    omegaDepth_              = 5.f;
    omegaNormals_            = 1.f;
    depthRejectionThreshold_ = 0.8f; // 0.25f;
    kernelChiThreshold_      = 1.f;
    entry_array_size_        = 0;

    camType_ = CameraType::Pinhole;
    cameraMatrix_.setIdentity();
    X_.setIdentity();
    sensorOffsetRotationInverse_.setIdentity();
    sensorOffsetInverse_.setIdentity();
    SX_.setIdentity();
    workspace_ = new Workspace();
  }

  void PhotometricFac::setFixedData(ImageSet* refData) {
    // propagate information about fixed/reference data to internal image set
    referenceData_               = refData;
    rows_                        = referenceData_->rows();
    cols_                        = referenceData_->cols();
    minDepth_                    = referenceData_->attributes_->min_depth;
    maxDepth_                    = referenceData_->attributes_->max_depth;
    cameraMatrix_                = referenceData_->sensor_->cameraMatrix();
    camType_                     = referenceData_->sensor_->cameraType();
    sensorOffsetInverse_         = referenceData_->sensor_->sensorOffset().inverse();
    sensorOffsetRotationInverse_ = sensorOffsetInverse_.linear();
    // referenceData_->matrix_.toDevice();
  }

  void PhotometricFac::setMovingData(const ImageSet& currData) {
    currData.toCloudDevice(&currData_);
  }

  void PhotometricFac::setTransform(const Eigen::Isometry3f& transform) {
    X_         = transform;
    SX_        = sensorOffsetInverse_ * X_;
    neg2rotSX_ = -2.f * SX_.linear();
  }

  std::pair<float, float> PhotometricFac::compute(const size_t max_iterations, const bool print_stats) {
    // checks if needed variables are correctly set
    assert((referenceData_->rows() != 0 || referenceData_->cols() != 0) &&
           "PhotometricFac::compute|level rows or columns set to zero");
    assert((referenceData_->max_depth != 0.f || referenceData_->min_depth != 0.f) &&
           "PhotometricFac::compute|level level max_depth or min_depth set to zero");

    std::pair<float, float> stats;
    for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
      setTransform(X_); // update offseted transformations
      computeProjections();
      stats = linearize();
    }

    // if (print_stats) {
    //   // std::cout << "photometric|compute time " << reg_time << " [ms] "
    //   //           << std::endl;
    //   std::cout << "photometric|residual " << stats.first << std::endl;
    //   std::cout << "photometric|inlier ratio " << stats.second << std::endl;
    // }
    return stats;
  }

} // namespace photo
