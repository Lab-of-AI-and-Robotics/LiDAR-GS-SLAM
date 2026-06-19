#include "image_set.cuh"

namespace photo {

  ImageSet::~ImageSet() {
  }

  ImageSet::ImageSet(const size_t rows, const size_t cols) {
    resize(rows, cols);
  }

  void ImageSet::resize(const size_t rows, const size_t cols) {
    matrix_.resize(rows, cols);
  }

  template <typename Matrix_>
  void applyPolicy(ImageEntry& entry, Matrix_&& m, FilterPolicy policy, float squared_threshold) {
    if (entry.masked())
      return;

    float n = m.squaredNorm();
    if (n < squared_threshold)
      return;

    switch (policy) {
      case Suppress:
        entry.setMasked(1);
        break;
      case Clamp:
        m *= sqrt(squared_threshold / n);
        break;
      default:;
    }
  }

  void ImageSet::updateDerivatives() {
    // these are for the normalization
    const float i2 = pow(attributes_->thresholds[Intensity], 2);
    const float d2 = pow(attributes_->thresholds[Depth], 2);
    const float n2 = pow(attributes_->thresholds[Normal], 2);

    const size_t rows = matrix_.rows();
    const size_t cols = matrix_.cols();
    // we start from 1st row
    for (size_t r = 1; r < rows - 1; ++r) {
      // fetch the row vectors
      // in the iteration below we start from the 1st column
      // so we increment the pointers by 1

      for (size_t c = 1; c < cols - 1; ++c) {
        ImageEntry& entry          = matrix_.at(r, c);
        const ImageEntry& entry_r0 = matrix_.at(r - 1, c);
        const ImageEntry& entry_r1 = matrix_.at(r + 1, c);
        const ImageEntry& entry_c0 = matrix_.at(r, c - 1);
        const ImageEntry& entry_c1 = matrix_.at(r, c + 1);

        // retrieve value
        const Vector5f& v_r0 = entry_r0.value;
        const Vector5f& v_r1 = entry_r1.value;
        const Vector5f& v_c0 = entry_c0.value;
        const Vector5f& v_c1 = entry_c1.value;

        // compute derivatives
        Matrix5_2f& derivatives = entry.derivatives;
        derivatives.col(1)      = .5 * v_r1 - .5 * v_r0;
        derivatives.col(0)      = .5 * v_c1 - .5 * v_c0;

        // here we ignore, clamp or suppress
        // the derivatives according to the selected policy
        applyPolicy(entry, derivatives.row(0), attributes_->policies[Intensity], i2);
        applyPolicy(entry, derivatives.row(1), attributes_->policies[Depth], d2);
        applyPolicy(entry, derivatives.block<3, 2>(2, 0), attributes_->policies[Normal], n2);
      }
    }
  }

  void ImageSet::getNormals(Matrixf3& normals) const {
    normals.resize(rows(), cols());
    for (size_t k = 0; k < matrix_.size(); ++k)
      normals.at(k) = matrix_.at(k).normal();
  }

  void ImageSet::growMask() {
    const int& radius = attributes_->mask_grow_radius;
    std::vector<int> ball_offsets;
    int r2 = pow(radius, 2);
    for (int r = -radius; r < radius + 1; ++r) {
      for (int c = -radius; c < radius + 1; ++c) {
        int idx = r * cols() + c;
        if ((r * r + c * c) <= r2) {
          ball_offsets.push_back(idx);
        }
      }
    }

    Matrixui8t mask(rows(), cols());
    for (size_t i = 0; i < mask.size(); ++i)
      mask[i] = matrix_.at(i).masked();
    for (size_t i = 0; i < mask.size(); ++i) {
      if (!mask[i])
        continue;
      for (auto offset : ball_offsets) {
        int target = offset + i;
        if (target < 0 || target >= (int) matrix_.size())
          continue;
        matrix_.at(target).setMasked(true);
      }
    }
  }

  static inline Eigen::Vector3f lift(const float f) {
    return Eigen::Vector3f(f, f, f);
  }

} // namespace photo
