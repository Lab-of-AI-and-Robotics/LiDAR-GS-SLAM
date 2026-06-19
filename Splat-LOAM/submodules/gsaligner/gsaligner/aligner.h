#pragma once
#include "registration/geometric.h"
#include "registration/photometric.cuh"
#include <torch/extension.h>

#include <tuple>

/**
 * @brief The following function should act as the main interface for the total
 * registration process. It takes the following inputs:
 * - ref_T_query: a torch tensor of shape (4, 4) containing the initial guess
 * - reference_depth: A torch tensor of shape (1, H, W) containing the depth
 * image of the reference frame.
 * - query_depth: A torch tensor of shape (1, H, W) containing the depth image
 * of the query frame
 * - reference_projmat: A torch tensor of shape (4, 4) containing the projection
 * matrix of the reference frame.
 * - Geometric:
 * - b_min: when a node is flatten than this param, propagate normal
 * - b_max: max size of kd leaves
 * - b_ratio: the increase factor of search radius needed in data association
 * - rho_ker: huber threshold in mad-icp
 * - iterations: number of iterations
 * - Photometric:
 * - omegaIntensity <- Discard for now
 * - omegaDepth
 * - omegaNormals <- Discard for now
 * - depthRejectionThreshold
 * - kernelChiThreshold
 * - minDepth
 * - maxDepth
 * - iterations
 * - intensity_derivative_threshold <- Discard for now
 * - depth_derivative_threshold
 * - normals_derivative_threshold <- Discard for now
 *
 * The function should return:
 * - ref_T_query: a torch tensor of shape (4, 4) containing the final
 * transformation
 * - fitness: a float containing the fitness of the registration (no. inliers /
 * no. points)
 * - rmse: a float containing the root mean squared error of the registration
 */

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
                                                              const int photo_iterations);

class GSAligner {
public:
  GSAligner(const int image_height_,
            const int image_width_,
            const float geom_b_min_,
            const float geom_b_max_,
            const float geom_b_ratio_,
            const float geom_rho_ker_,
            const int geom_iterations_,
            const float photo_omega_depth_,
            const float photo_depth_rejection_threshold_,
            const float photo_rho_ker_,
            const float photo_min_depth_,
            const float photo_max_depth_,
            const int photo_iterations_) :
    image_height(image_height_),
    image_width(image_width_),
    geom_b_min(geom_b_min_),
    geom_b_max(geom_b_max_),
    geom_b_ratio(geom_b_ratio_),
    geom_rho_ker(geom_rho_ker_),
    geom_iterations(geom_iterations_),
    photo_omega_depth(photo_omega_depth_),
    photo_depth_rejection_threshold(photo_depth_rejection_threshold_),
    photo_rho_ker(photo_rho_ker_),
    photo_min_depth(photo_min_depth_),
    photo_max_depth(photo_max_depth_),
    photo_iterations(photo_iterations_),
    mad_icp(new GeometricFac(geom_b_max_, geom_rho_ker_, geom_b_ratio_, 1)),
    photo_attributes(std::make_shared<photo::ImageAttributes>()),
    ref_sensor(std::make_shared<photo::Sensor>()),
    query_sensor(std::make_shared<photo::Sensor>())

  {
    photo_attributes->min_depth                  = photo_min_depth;
    photo_attributes->max_depth                  = photo_max_depth;
    photo_attributes->policies[photo::Intensity] = (photo::FilterPolicy) 0; // Ignore intensity
    photo_attributes->policies[photo::Depth]     = (photo::FilterPolicy) 2; // Use depth
    photo_attributes->thresholds[photo::Depth]   = 0.5;
    photo_attributes->policies[photo::Normal]    = (photo::FilterPolicy) 0; // Ignore normal
    photo_attributes->mask_grow_radius           = 1.0;

    ref_sensor->rows()           = image_height;
    ref_sensor->cols()           = image_width;
    ref_sensor->sensorOffset()   = Eigen::Isometry3f::Identity();
    ref_sensor->cameraType()     = photo::CameraType::Spherical;
    query_sensor->rows()         = image_height;
    query_sensor->cols()         = image_width;
    query_sensor->sensorOffset() = Eigen::Isometry3f::Identity();
    query_sensor->cameraType()   = photo::CameraType::Spherical;
    ref_sensor->setDepthScale(1.0);
    query_sensor->setDepthScale(1.0);

    photo_fact.setOmegaDepth(photo_omega_depth);
    photo_fact.setDepthRejectionThreshold(photo_depth_rejection_threshold);
    photo_fact.setKernelChiThreshold(photo_rho_ker);
  }

  void setReference(const torch::Tensor& ref_depth_, const torch::Tensor& ref_cloud_, const torch::Tensor& ref_projmat_);
  void setQuery(const torch::Tensor& query_depth_, const torch::Tensor& query_cloud_, const torch::Tensor& query_projmat_);

  std::tuple<torch::Tensor, float, float> alignGeometric(const torch::Tensor& ref_T_query_ig);
  std::tuple<torch::Tensor, float, float> alignPhotometric(const torch::Tensor& ref_T_query_ig);

protected:
  const int image_height;
  const int image_width;
  const float geom_b_min;
  const float geom_b_max;
  const float geom_b_ratio;
  const float geom_rho_ker;
  const int geom_iterations;
  const float photo_omega_depth;
  const float photo_depth_rejection_threshold;
  const float photo_rho_ker;
  const float photo_min_depth;
  const float photo_max_depth;
  const int photo_iterations;

  std::unique_ptr<GeometricFac> mad_icp;
  std::vector<Eigen::Vector3d> ref_cloud;
  std::vector<Eigen::Vector3d> query_cloud;
  std::unique_ptr<MADtree> geom_ref_tree;
  std::unique_ptr<MADtree> geom_query_tree;
  LeafList geom_query_leaves;

  // Photometric
  photo::Image ref_image, query_image;
  photo::SensorPtr ref_sensor, query_sensor;
  photo::ImageAttributesPtr photo_attributes;
  photo::ImageSet ref_set, query_set;
  photo::PhotometricFac photo_fact;
};