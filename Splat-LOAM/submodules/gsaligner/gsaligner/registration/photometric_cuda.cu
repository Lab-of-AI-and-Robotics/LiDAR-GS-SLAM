#include "photometric.cuh"
#include <tools/sum_reduce.cu>

namespace photo {

  ///////////////////////////////////////////////// PROJECTIONS KERNELS /////////////////////////////////////////////////

  __global__ void project_kernel(Workspace* workspace,
                                 const MatrixCloud* cloud,
                                 const Image* ref_image,
                                 const Eigen::Isometry3f SX,
                                 const CameraType cam_type,
                                 const Eigen::Matrix3f cam_mat,
                                 const float min_depth,
                                 const float max_depth,
                                 const bool is_depth_buffer) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= cloud->size())
      return;

    const auto full_point                   = cloud->at<1>(tid);
    const Eigen::Vector3f point             = full_point.coordinates;
    const Eigen::Vector3f normal            = full_point.normal;
    const float intensity                   = full_point.intensity;
    const Eigen::Vector3f transformed_point = SX * point;

    float depth                  = 0.f;
    Eigen::Vector3f camera_point = Eigen::Vector3f::Zero();
    Eigen::Vector2f image_point  = Eigen::Vector2f::Zero();

    const bool is_good = project(image_point, camera_point, depth, transformed_point, cam_type, cam_mat, min_depth, max_depth);

    if (!is_good)
      return;

    // equivalent of cvRound
    const int irow = (int) (image_point.y() + (image_point.y() >= 0 ? 0.5f : -0.5f));
    const int icol = (int) (image_point.x() + (image_point.x() >= 0 ? 0.5f : -0.5f));

    if (!workspace->inside(irow, icol)) {
      return;
    }

    if (ref_image->at<1>(irow, icol).masked()) {
      return;
    }

    WorkspaceEntry& entry = workspace->at<1>(irow, icol);

    if (is_depth_buffer) {
      // depth buffer implemented as comparison between two uint64
      atomicMin(&(entry.depthIdx), packIntFloat(tid, depth));
      return;
    }

    unsigned long long candidate_depth_idx = packIntFloat(tid, depth);
    if (candidate_depth_idx != entry.depthIdx)
      return;

    // unpack depth and tread id from single depth_idx variable
    unpackIntFloat(entry.index, depth, candidate_depth_idx);

    const Eigen::Vector3f rn = SX.linear() * normal;
    entry.prediction << intensity, depth, rn.x(), rn.y(), rn.z();
    entry.point            = point;
    entry.normal           = normal;
    entry.transformedPoint = transformed_point;
    entry.cameraPoint      = camera_point;
    entry.imagePoint       = image_point;
    entry.chi              = 0.f;
  }

  void PhotometricFac::computeProjections() {
    // resize workspace based on rows and columns
    workspace_->resize(rows_, cols_);
    // initialize workspace (only in device)
    workspace_->fill(WorkspaceEntry(), true);

    // call project_kernel with is_depth_buffer_ set to true
    project_kernel<<<workspace_->nBlocks(), workspace_->nThreads()>>>(workspace_->deviceInstance(),
                                                                      currData_.deviceInstance(),
                                                                      referenceData_->matrix_.deviceInstance(),
                                                                      SX_,
                                                                      camType_,
                                                                      cameraMatrix_,
                                                                      minDepth_,
                                                                      maxDepth_,
                                                                      true);
    CUDA_CHECK(cudaDeviceSynchronize());

    // call project_kernel with is_depth_buffer_ set to false
    project_kernel<<<workspace_->nBlocks(), workspace_->nThreads()>>>(workspace_->deviceInstance(),
                                                                      currData_.deviceInstance(),
                                                                      referenceData_->matrix_.deviceInstance(),
                                                                      SX_,
                                                                      camType_,
                                                                      cameraMatrix_,
                                                                      minDepth_,
                                                                      maxDepth_,
                                                                      false);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  __device__ PointStatusFlag errorAndJacobian(Vector5f& e,
                                              Matrix5_6f& J,
                                              WorkspaceEntry& entry,
                                              const Image* image,
                                              const Eigen::Isometry3f& SX,
                                              const Eigen::Matrix3f& neg2rotSX,
                                              const Eigen::Matrix3f& cam_matrix,
                                              const CameraType& cam_type,
                                              const float weight_intensity,
                                              const float weight_depth,
                                              const float weight_normals,
                                              const float d_threshold,
                                              bool chi_only) {
    PointStatusFlag status                  = Good;
    const float z                           = entry.depth();
    const float iz                          = 1.f / z;
    const Eigen::Vector3f point             = entry.point;
    const Eigen::Vector3f normal            = entry.normal;
    const Eigen::Vector3f transformed_point = entry.transformedPoint;
    const Eigen::Vector3f camera_point      = entry.cameraPoint;
    const Eigen::Vector2f image_point       = entry.imagePoint;

    Vector5f measurement;
    Matrix5_2f image_derivatives;
    // todo level ptr
    const bool ok = getSubPixel(measurement, image_derivatives, image, image_point);
    if (!ok) {
      // printf("suca!\t");
      return Masked;
    }

    // in error put the difference between prediction and measurement
    e           = entry.prediction - measurement;
    entry.error = e;
    e(0) *= weight_intensity;
    e(1) *= weight_depth;
    e.tail(3) *= weight_normals;

    // if depth error is to big we drop
    const float depth_error = e(1) * e(1);
    if (depth_error > d_threshold) {
      // printf("depth not ok! [img.x=%f img.y=%f depth=%f derr=%f t=%f]\n",
      //        image_point.x(),
      //        image_point.y(),
      //        z,
      //        depth_error,
      //        d_threshold);
      // printf("[point=[%f %f %f] tpoint=[%f %f %f] [row=%f col=%f pred=%f meas=%f err=%f]\n",
      //        point.x(),
      //        point.y(),
      //        point.z(),
      //        transformed_point.x(),
      //        transformed_point.y(),
      //        transformed_point.z(),
      //        image_point.y(),
      //        image_point.x(),
      //        entry.prediction(1),
      //        measurement(1),
      //        e(1));
      return DepthError;
    }
    if (chi_only)
      return status;

    J.setZero();

    // compute the pose jacobian, including sensor offset
    Matrix3_6f J_icp        = Matrix3_6f::Zero();
    J_icp.block<3, 3>(0, 0) = SX.linear();
    J_icp.block<3, 3>(0, 3) = neg2rotSX * lie::skew(point);

    // extract values from hom for readability
    const float iz2 = iz * iz;

    // extract the values from camera matrix
    const float fx = cam_matrix(0, 0);
    const float fy = cam_matrix(1, 1);
    const float cx = cam_matrix(0, 2);
    const float cy = cam_matrix(1, 2);

    // computes J_hom*K explicitly to avoid matrix multiplication and stores it in J_proj
    Matrix2_3f J_proj = Matrix2_3f::Zero();

    switch (cam_type) {
      case CameraType::Pinhole:
        // fill the left  and the right 2x3 blocks of J_proj with J_hom*K
        J_proj(0, 0) = fx * iz;
        J_proj(0, 2) = cx * iz - camera_point.x() * iz2;
        J_proj(1, 1) = fy * iz;
        J_proj(1, 2) = cy * iz - camera_point.y() * iz2;

        // add the jacobian of depth prediction to row 1.
        J.row(1) = J_icp.row(2);

        break;
      case CameraType::Spherical: {
        const float ir    = iz;
        const float ir2   = iz2;
        const float rxy2  = transformed_point.head(2).squaredNorm();
        const float irxy2 = 1. / rxy2;
        const float rxy   = sqrt(rxy2);
        const float irxy  = 1. / rxy;

        J_proj << -fx * transformed_point.y() * irxy2, // 1st row
          fx * transformed_point.x() * irxy2, 0,
          -fy * transformed_point.x() * transformed_point.z() * irxy * ir2, // 2nd row
          -fy * transformed_point.y() * transformed_point.z() * irxy * ir2, fy * rxy * ir2;

        Matrix1_3f J_depth; // jacobian of range(x,y,z)
        J_depth << transformed_point.x() * ir, transformed_point.y() * ir, transformed_point.z() * ir;

        // add the jacobian of range/depth prediction to row 1
        J.row(1) = J_depth * J_icp;

      } break;
    }

    // chain rule to get the jacobian
    J -= image_derivatives * J_proj * J_icp;

    // including normals
    J.block<3, 3>(2, 3) += neg2rotSX * lie::skew(normal);

    // Information mat is diagonal matrix
    // to avoid multiplications we premultiply the rows of J by sqrt of diag
    // elements

    J.row(0) *= weight_intensity;
    J.row(1) *= weight_depth;
    J.block<3, 2>(2, 0) *= weight_normals;
    return status;
  }

  __global__ void linearize_kernel(LinearSystemEntry* ls_entry, // dst
                                   Workspace* workspace,        // src-dst
                                   const Image* image,
                                   const Eigen::Isometry3f SX,
                                   const Eigen::Matrix3f neg2rotSX,
                                   const Eigen::Matrix3f cam_matrix,
                                   const CameraType cam_type,
                                   const float e_threshold,
                                   const double scaling,
                                   const float weight_intensity,
                                   const float weight_depth,
                                   const float weight_normals,
                                   const float d_threshold,
                                   const bool chi_only = false) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= workspace->size())
      return;

    Vector5f e   = Vector5f::Zero();
    Matrix5_6f J = Matrix5_6f::Zero();

    WorkspaceEntry& ws_entry = workspace->at<1>(tid);
    const int idx            = ws_entry.index;

    if (idx == -1)
      return;

    PointStatusFlag status = ws_entry.status;
    if (status != Good) {
      return;
    }

    status = errorAndJacobian(e,
                              J,
                              ws_entry,
                              image,
                              SX,
                              neg2rotSX,
                              cam_matrix,
                              cam_type,
                              weight_intensity,
                              weight_depth,
                              weight_normals,
                              d_threshold,
                              chi_only);
    if (status != Good) {
      return;
    }
    // from now will be evaluated later
    ls_entry[idx].is_good = 1;
    const float chi       = e.dot(e);

    ls_entry[idx].chi = ws_entry.chi = chi;
    float lambda                     = 1.f;
    if (chi > e_threshold) {
      lambda = sqrt(e_threshold / chi);
    } else {
      ls_entry[idx].is_inlier = 1;
    }
    if (!chi_only) {
      // here will add all good contribution of pixels
      // outliers contribution will be zero in the sum reduction

      // using double the precision for accumulation
      Matrix6d tmp_full_H = Matrix6d::Zero();
      computeAtxA(tmp_full_H, J, lambda);
      tmp_full_H *= scaling;

      ls_entry[idx].upperH = mat6DUpperTriangularToVector(tmp_full_H);
      ls_entry[idx].b      = (J.transpose() * e * lambda).cast<double>() * scaling;
    }
  }

  std::pair<float, float> PhotometricFac::linearize() {
    const float weight_intensity = std::sqrt(omegaIntensity_);
    const float weight_depth     = std::sqrt(omegaDepth_);
    const float weight_normals   = std::sqrt(omegaNormals_);

    const double scaling = 1.0 / workspace_->size();

    // compute workspace only when is needed
    if (entry_array_size_ != workspace_->size()) {
      if (entry_array_) {
        CUDA_CHECK(cudaFree(entry_array_));
        entry_array_ = nullptr;
      }
      if (workspace_->size())
        CUDA_CHECK(cudaMalloc((void**) &entry_array_, sizeof(LinearSystemEntry) * workspace_->size()));
      entry_array_size_ = workspace_->size();
    }

    LinearSystemEntry sum;
    if (!workspace_->empty()) {
      // clear linear system entry buffer
      LinearSystemEntry zeroval;
      fill_kernel<<<workspace_->nBlocks(), workspace_->nThreads()>>>(entry_array_, zeroval, workspace_->size());
      CUDA_CHECK(cudaDeviceSynchronize());

      // linearize get solver entry from two pyramids
      linearize_kernel<<<workspace_->nBlocks(), workspace_->nThreads()>>>(entry_array_,
                                                                          workspace_->deviceInstance(),
                                                                          referenceData_->matrix_.deviceInstance(),
                                                                          SX_,
                                                                          neg2rotSX_,
                                                                          cameraMatrix_,
                                                                          camType_,
                                                                          kernelChiThreshold_,
                                                                          scaling,
                                                                          weight_intensity,
                                                                          weight_depth,
                                                                          weight_normals,
                                                                          depthRejectionThreshold_);
      CUDA_CHECK(cudaDeviceSynchronize());

      // reduce sum
      const int num_threads    = BLOCKSIZE; // macro optimized for LinearSystemEntry
      const int num_blocks     = (workspace_->size() + num_threads - 1) / num_threads;
      const int required_shmem = num_threads * sizeof(LinearSystemEntry);
      LinearSystemEntry* dsum_block;
      CUDA_CHECK(cudaMalloc((void**) &dsum_block, sizeof(LinearSystemEntry) * num_blocks));
      sum_reduce_wrapper(dsum_block, entry_array_, workspace_->size(), num_blocks, num_threads);

      // copy to host and do last reduction, useless to evocate the gpu for stupid problem
      LinearSystemEntry* hsum_block = new LinearSystemEntry[num_blocks];
      CUDA_CHECK(cudaMemcpy(hsum_block, dsum_block, num_blocks * sizeof(LinearSystemEntry), cudaMemcpyDeviceToHost));
      // sum latest part, buffer equal to num of blocks
      for (int i = 0; i < num_blocks; i++) {
        sum += hsum_block[i];
      }

      // free mem
      delete hsum_block;
      CUDA_CHECK(cudaFree(dsum_block));
    }

    // get total linear system entry H, b and tot error
    Matrix6d tot_H = vectorToMat6DUpperTriangular(sum.upperH);
    copyLowerTriangleUp(tot_H);
    const Vector6d& tot_b    = sum.b;
    const float tot_chi      = sum.chi;
    const size_t num_inliers = sum.is_inlier;
    const size_t num_good    = sum.is_good;

    // if num good is 0 fucks chi up, if we don't have any inliers is
    // likely num good is very low
    if (!num_good || !num_inliers) {
      return std::pair(tot_chi, float(num_inliers) / float(rows_ * cols_));
    }
    const float residual     = tot_chi / num_good;
    const float inlier_ratio = float(num_inliers) / float(rows_ * cols_);
    const Vector6d dx        = tot_H.ldlt().solve(-tot_b);
    Eigen::Isometry3d dX     = Eigen::Isometry3d::Identity();
    dX.linear()              = lie::expMapSO3(dx.tail(3)); // TODO maybe stick to quaternion?
    dX.translation()         = dx.head(3);
    X_                       = X_ * dX.cast<float>();

    return std::pair(residual, inlier_ratio);
  }
} // namespace photo
