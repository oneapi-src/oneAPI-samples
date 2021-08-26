//==============================================================
// Copyright © 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace pointpillars {

/**
 * Anchor representation
 */
struct Anchor {
  /**
  * @brief Constructor
  * @param[in] _x Size along x
  * @param[in] _y Size along y
  * @param[in] _z Size along z
  */
  Anchor(float _x, float _y, float _z) : x(_x), y(_y), z(_z), dz(0.){};

  /**
  * @brief Constructor
  * @param[in] _x Size along x
  * @param[in] _y Size along y
  * @param[in] _z Size along z
  * @param[in] _dz Position in z
  */
  Anchor(float _x, float _y, float _z, float _dz) : x(_x), y(_y), z(_z), dz(_dz){};

  float x{1.f};
  float y{1.f};
  float z{1.f};
  float dz{0.f};
};

struct AnchorGridConfig {
  float min_x_range{0.0f};  // defines the area covered
  float max_x_range{1.0f};  // defines the area covered
  float min_y_range{0.0f};  // defines the area covered
  float max_y_range{1.0f};  // defines the area covered
  float min_z_range{0.0f};  // defines the area covered
  float max_z_range{1.0f};  // defines the area covered

  float x_stride{0.01f};  // spacing between anchors along x
  float y_stride{0.01f};  // spacing between anchors along y

  std::vector<Anchor> anchors = {Anchor(1.0f, 2.0f, 1.5f)};
  std::vector<float> rotations = {0.f, M_PI_2};  // The set of rotations to in which the anchors should be generated
};

struct PointPillarsConfig {
  std::string pfe_model_file{"pfe"};
  std::string rpn_model_file{"rpn"};
  float min_x_range{0.f};      // defines the area covered by the algorithm
  float max_x_range{69.12f};   // defines the area covered by the algorithm
  float min_y_range{-39.68f};  // defines the area covered by the algorithm
  float max_y_range{39.68f};   // defines the area covered by the algorithm
  float min_z_range{-3.0f};    // defines the area covered by the algorithm
  float max_z_range{1.0f};     // defines the area covered by the algorithm
  float rpn_scale{0.5f};       // The scaling factor that the RPN is applying = 1 / (final convolution stride)
  float pillar_x_size{0.16f};  // pillar voxelization size along X
  float pillar_y_size{0.16f};  // pillar voxelization size along Y
  float pillar_z_size{4.0f};   // pillar voxelization size along Z
  float x_stride{0.32f};       // spacing between pillars along x
  float y_stride{0.32f};       // spacing between pillars along y
  std::size_t max_num_pillars{12000};
  std::size_t num_classes{1};
  std::vector<Anchor> anchors = {Anchor(1.6f, 3.9f, 1.56f)};
  std::vector<std::string> classes = {"Car"};
  std::size_t max_num_points_per_pillar{100};
  std::size_t pillar_features{64};
  std::size_t grid_x_size{432};  // (max_x_range - min_x_range) / pillar_x_size
  std::size_t grid_y_size{496};  // (max_y_range - min_y_range) / pillar_y_size
  std::size_t grid_z_size{1};    // (max_z_range - min_z_range) / pillar_z_size
};
}  // namespace pointpillars
