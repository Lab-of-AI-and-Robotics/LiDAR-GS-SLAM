#include "aligner.h"

#include "aligner_utils.cuh"
#include "registration/geometric.h"
#include <Eigen/Core>
#include <iterator>
#include <torch/extension.h>
#include <vector>

#define CHECK_INPUT(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
// TODO: Controllare
constexpr int MAX_PARALLEL_LEVELS = 4;

std::vector<Eigen::Vector3f> TensorToVector3f(const torch::Tensor& tensor) {
  std::vector<Eigen::Vector3f> vec(tensor.size(0));
  for (int i = 0; i < tensor.size(0); ++i) {
    vec[i] = Eigen::Map<Eigen::Vector3f>(tensor[i].data<float>());
  }
  return vec;
}
// std::vector<Eigen::Vector3d> TensorToVector3d(const torch::Tensor& tensor) {
//   std::vector<Eigen::Vector3d> vec(tensor.size(0));
//   for (int i = 0; i < tensor.size(0); ++i) {
//     vec[i] = (Eigen::Map<Eigen::Vector3f>(tensor[i].data<float>())).cast<double>();
//   }
//   return vec;
// }

std::vector<Eigen::Vector3d> TensorToVector3d(const torch::Tensor& tensor) {
  // Copy the data, not the best approach, but mad doesn't support Eigen::Map
  std::vector<Eigen::Vector3d> vec;
  vec.reserve(tensor.size(0));
  for (int i = 0; i < tensor.size(0); ++i) {
    vec.emplace_back(tensor[i].data<double>());
  }
  return vec;
}

std::vector<Eigen::Map<Eigen::Vector3d>> TensorToVector3dMap(const torch::Tensor& tensor) {
  std::vector<Eigen::Map<Eigen::Vector3d>> vec;
  vec.reserve(tensor.size(0));
  for (int i = 0; i < tensor.size(0); ++i) {
    vec.emplace_back(tensor[i].data<double>());
  }
  return vec;
}

std::tuple<torch::Tensor, float, float> RegistrationGeomPhoto(const torch::Tensor& ref_T_query_init,
                                                              const torch::Tensor& reference_depth,
                                                              const torch::Tensor& query_depth,
                                                              const torch::Tensor& reference_cloud,
                                                              const torch::Tensor& query_cloud,
                                                              const torch::Tensor& reference_projmat,
                                                              const int image_height,
                                                              const int image_width,
                                                              const float geom_b_min,
                                                              const float geom_b_max,
                                                              const float geom_b_ratio,
                                                              const float geom_rho_ker,
                                                              const int geom_iterations,
                                                              const float photo_omega_depth,
                                                              const float photo_depth_rejection_threshold,
                                                              const float photo_rho_ker,
                                                              const float photo_min_depth,
                                                              const float photo_max_depth,
                                                              const int photo_iterations) {
  // const int H = image_height;
  // const int W = image_width;

  CHECK_INPUT(ref_T_query_init);
  CHECK_INPUT(reference_depth);
  CHECK_INPUT(query_depth);
  CHECK_INPUT_CPU(reference_cloud);
  CHECK_INPUT_CPU(query_cloud);
  CHECK_INPUT(reference_projmat);

  if (reference_depth.ndimension() != 3) {
    AT_ERROR("reference_depth must be a (1xHxW) tensor");
  }
  if (query_depth.ndimension() != 3) {
    AT_ERROR("reference_depth must be a (1xHxW) tensor");
  }

  if (reference_cloud.ndimension() != 2 || reference_cloud.size(1) != 3) {
    AT_ERROR("reference_cloud must be a (Nx3) tensor");
  }

  if (query_cloud.ndimension() != 2 || query_cloud.size(1) != 3) {
    AT_ERROR("reference_cloud must be a (Nx3) tensor");
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  torch::Tensor ref_T_query = torch::eye(4, options);
  float align_fitness       = 0.0;
  float align_rmse          = 0.0;

  /** This is the merging point between torch and gsalign
   * To do that, we now have to carefully pass shit to the appropriate stuff
   */

  // Geometric
  auto std_ref_cloud   = TensorToVector3d(reference_cloud);
  auto std_query_cloud = TensorToVector3d(query_cloud);
  // TODO: get T from ref_T_query_init
  auto T = Eigen::Matrix4d::Identity();

  std::unique_ptr<MADtree> ref_tree;
  ref_tree.reset(new MADtree(&std_ref_cloud,
                             std_ref_cloud.begin(),
                             std_ref_cloud.end(),
                             geom_b_max,
                             geom_b_min,
                             0,
                             MAX_PARALLEL_LEVELS,
                             nullptr,
                             nullptr));

  std::unique_ptr<MADtree> query_tree;
  query_tree.reset(new MADtree(&std_query_cloud,
                               std_query_cloud.begin(),
                               std_query_cloud.end(),
                               geom_b_max,
                               geom_b_min,
                               0,
                               MAX_PARALLEL_LEVELS,
                               nullptr,
                               nullptr));
  LeafList query_leaves_;
  query_tree->getLeafs(std::back_insert_iterator<LeafList>(query_leaves_));

  std::unique_ptr<GeometricFac> mad_icp_;

  mad_icp_.reset(new GeometricFac(geom_b_max, geom_rho_ker, geom_b_ratio, 1));
  mad_icp_->setMoving(query_leaves_);
  // make initial guess in right format
  Eigen::Isometry3d dX = Eigen::Isometry3d::Identity();
  dX.linear()          = T.block<3, 3>(0, 0);
  dX.translation()     = T.block<3, 1>(0, 3);
  mad_icp_->init(dX);

  float icp_time = 0;
  struct timeval icp_start, icp_end, icp_delta;
  gettimeofday(&icp_start, nullptr);

  // icp loop
  for (size_t icp_iteration = 0; icp_iteration < geom_iterations; ++icp_iteration) {
    if (icp_iteration == geom_iterations - 1) {
      for (MADtree* l : query_leaves_) {
        l->matched_ = false;
      }
    }
    mad_icp_->resetAdders();
    mad_icp_->update(ref_tree.get());
    mad_icp_->updateState();
  }

  gettimeofday(&icp_end, nullptr);
  timersub(&icp_end, &icp_start, &icp_delta);
  icp_time = float(icp_delta.tv_sec) * 1000. + 1e-3 * icp_delta.tv_usec;

  // if (print_stats) {
  int matched_leaves = 0;
  for (MADtree* l : query_leaves_) {
    if (l->matched_) {
      matched_leaves++;
    }
  }
  const double inliers_ratio = double(matched_leaves) / double(query_leaves_.size());
  std::cout << "geometricFac|compute time " << icp_time << " [ms] " << std::endl;
  std::cout << "geometricFac|inliers ratio " << inliers_ratio << std::endl;
  std::cout << "--geometricFac|matched leaves " << matched_leaves << std::endl;
  std::cout << "--geometricFac|total num leaves " << query_leaves_.size() << std::endl;
  std::cout << mad_icp_->X_.matrix() << std::endl;
  // }

  return std::make_tuple(ref_T_query, align_fitness, align_rmse);
}

void GSAligner::setReference(const torch::Tensor& ref_depth_,
                             const torch::Tensor& ref_cloud_,
                             const torch::Tensor& ref_projmat_) {
  /**
   * @brief
   * - Verify that ref_depth_ and ref_projmat_ are CUDA tensors, respectively of shape (1, H, W) and (3, 4)
   * - Verify that ref_cloud_ is a CPU torch::KDouble tensor of shape (N, 3)
   * - Convert ref_cloud_ to a std::vector<Eigen::Vector3d>
   * - - Concurrently, convert ref_depth_ into a DualMatrix and, ref_projmat_ into the appropriate class
   * - Create a MADtree from the ref_cloud_ and store it in ref_tree
   * - Create a GeometricFac object and store it in mad_icp
   */
  CHECK_INPUT(ref_depth_);
  CHECK_INPUT_CPU(ref_cloud_);
  CHECK_INPUT(ref_projmat_);

  if (ref_depth_.ndimension() != 3) {
    AT_ERROR("ref_depth_ must be a (1xHxW) tensor");
  }
  if (ref_cloud_.ndimension() != 2 || ref_cloud_.size(1) != 3) {
    AT_ERROR("ref_cloud_ must be a (Nx3) tensor");
  }
  if (ref_cloud_.dtype() != torch::kDouble) {
    AT_ERROR("ref_cloud_ must be of type torch::kDouble (or torch::kFloat64)");
  }
  ref_image = photo::Image(image_height, image_width);
  init_photo_image_from_buffer(ref_depth_, ref_image, image_height, image_width); // TODO: ref_image.toHost() if segfault occurs
  // ref_image.toHost(); // We don't want to do this (unnecessary copy to CPU memory)
  auto ref_projmat = ref_projmat_.to(torch::kCPU);
  Eigen::Matrix4f fullProjMatrix =
    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::ColMajor>>(ref_projmat.contiguous().data<float>());
  ref_sensor->cameraMatrix() = fullProjMatrix.block<3, 3>(0, 0);

  ref_set.matrix_     = ref_image; // TODO: Check if this is correct
  ref_set.attributes_ = photo_attributes;
  ref_set.sensor_     = ref_sensor;

  ref_set.growMaskDevice();
  ref_set.updateDerivativesDevice();

  photo_fact.setFixedData(&ref_set);

  ref_cloud = TensorToVector3d(ref_cloud_);
  geom_ref_tree.reset(new MADtree(
    &ref_cloud, ref_cloud.begin(), ref_cloud.end(), geom_b_max, geom_b_min, 0, MAX_PARALLEL_LEVELS, nullptr, nullptr));
}
void GSAligner::setQuery(const torch::Tensor& query_depth_,
                         const torch::Tensor& query_cloud_,
                         const torch::Tensor& query_projmat_) {
  /**
   * @brief
   * - Verify that query_depth_ is a CUDA tensor of shape (1, H, W)
   * - Verify that query_cloud_ is a CPU torch::KDouble tensor of shape (N, 3)
   * - Convert query_cloud_ to a std::vector<Eigen::Vector3d>
   * - - Concurrently, convert query_depth_ into a ImageSet
   * - Create a MADtree from the query_cloud_ and store it in query_tree
   * - Extract the LeafList from query_tree and store it in query_leaves
   */
  CHECK_INPUT(query_depth_);
  CHECK_INPUT(query_projmat_);
  CHECK_INPUT_CPU(query_cloud_);

  if (query_depth_.ndimension() != 3) {
    AT_ERROR("query_depth_ must be a (1xHxW) tensor");
  }
  if (query_cloud_.ndimension() != 2 || query_cloud_.size(1) != 3) {
    AT_ERROR("query_cloud_ must be a (Nx3) tensor");
  }
  if (query_cloud_.dtype() != torch::kDouble) {
    AT_ERROR("query_cloud_ must be of type torch::kDouble (or torch::kFloat64)");
  }
  // Photometric
  // TODO: Here we should launch kernels to concurrently instantiate DualMatrices and ref_projmat
  query_image = photo::Image(image_height, image_width);
  init_photo_image_from_buffer(
    query_depth_, query_image, image_height, image_width); // TODO: ref_image.toHost() if segfault occurs
  auto projmat                   = query_projmat_.to(torch::kCPU);
  Eigen::Matrix4f fullProjMatrix = Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::ColMajor>>(projmat.contiguous().data<float>());
  query_sensor->cameraMatrix()   = fullProjMatrix.block<3, 3>(0, 0);

  query_set.matrix_     = query_image; // TODO: Check if this is correct
  query_set.attributes_ = photo_attributes;
  // query_set.sensor_     = query_sensor;
  query_set.sensor_ = ref_sensor;

  photo_fact.setMovingData(query_set);

  // Geometric

  query_cloud = TensorToVector3d(query_cloud_);
  geom_query_tree.reset(new MADtree(
    &query_cloud, query_cloud.begin(), query_cloud.end(), geom_b_max, geom_b_min, 0, MAX_PARALLEL_LEVELS, nullptr, nullptr));
  geom_query_leaves.clear();
  geom_query_tree->getLeafs(std::back_insert_iterator<LeafList>(geom_query_leaves));

  mad_icp->setMoving(geom_query_leaves);
}

std::tuple<torch::Tensor, float, float> GSAligner::alignGeometric(const torch::Tensor& ref_T_query_ig) {
  /**
   * @brief
   * - set moving leaves in mad_icp
   * - make initial guess in right format
   * - run icp loop
   */
  CHECK_INPUT_CPU(ref_T_query_ig);
  if (ref_T_query_ig.ndimension() != 2 || ref_T_query_ig.size(0) != 4 || ref_T_query_ig.size(1) != 4) {
    AT_ERROR("ref_T_query_ig must be a (4x4) tensor");
  }
  if (ref_T_query_ig.dtype() != torch::kDouble) {
    AT_ERROR("ref_T_query_ig must be of type torch::kDouble (or torch::kFloat64)");
  }
  if (geom_ref_tree == nullptr || geom_query_tree == nullptr) {
    AT_ERROR("Reference and Query trees must be set before calling alignGeometric");
  }

  const Eigen::Matrix4d& T = Eigen::Map<Eigen::Matrix4d>(ref_T_query_ig.data<double>());
  Eigen::Isometry3d dX     = Eigen::Isometry3d::Identity();
  dX.linear()              = T.block<3, 3>(0, 0);
  dX.translation()         = T.block<3, 1>(0, 3);
  mad_icp->init(dX);

  for (size_t i = 0; i < geom_iterations; ++i) {
    if (i == geom_iterations - 1) {
      for (MADtree* l : geom_query_leaves) {
        l->matched_ = false;
      }
    }
    mad_icp->resetAdders();
    mad_icp->update(geom_ref_tree.get());
    mad_icp->updateState();
  }

  size_t matched_leaves = 0;
  for (MADtree* l : geom_query_leaves) {
    if (l->matched_) {
      matched_leaves++;
    }
  }

  float fitness = double(matched_leaves) / double(geom_query_leaves.size());
  float inlier_rmse =
    0.0; // TODO: Extract determinant of the inverse of the H matrix (see
         // https://github.com/rvp-group/mad-icp/blob/cbf53a90ec7d4cb525618a11598bbb5e471a7e79/mad_icp/src/odometry/pipeline.cpp#L223
         // )
  torch::Tensor res = torch::from_blob(mad_icp->X_.matrix().data(), {4, 4}, torch::kDouble).clone().transpose(0, 1);

  return std::make_tuple(res, fitness, inlier_rmse);
}

std::tuple<torch::Tensor, float, float> GSAligner::alignPhotometric(const torch::Tensor& ref_T_query_ig) {
  CHECK_INPUT_CPU(ref_T_query_ig);

  Eigen::Matrix4f ig_eigen = Eigen::Map<Eigen::Matrix4f, Eigen::RowMajor>(ref_T_query_ig.contiguous().data<float>()).transpose();
  Eigen::Isometry3f ig     = Eigen::Isometry3f::Identity();
  ig.linear()              = ig_eigen.block<3, 3>(0, 0);
  ig.translation()         = ig_eigen.block<3, 1>(0, 3);
  photo_fact.setTransform(ig);
  const auto stats        = photo_fact.compute(photo_iterations, true);
  Eigen::Isometry3f res   = photo_fact.transform();
  torch::Tensor res_torch = torch::from_blob(res.matrix().data(), {4, 4}, torch::kFloat32).clone().transpose(0, 1);
  return std::make_tuple(res_torch, stats.second, stats.first);
}