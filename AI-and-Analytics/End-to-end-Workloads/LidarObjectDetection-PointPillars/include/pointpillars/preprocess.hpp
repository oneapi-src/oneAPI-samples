/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 * Copyright (c) 2019-2021 Intel Corporation (oneAPI modifications)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <CL/sycl.hpp>

namespace pointpillars {

/**
 * PointPillar's PreProcessing
 *
 * Convert 3D point cloud data into 2D-grid/pillar form
 * to be able to feed it into the PillarFeatureNetwork
 */
class PreProcess {
 private:
  // initialzer list
  const int max_num_pillars_;
  const int max_num_points_per_pillar_;
  const int grid_x_size_;
  const int grid_y_size_;
  const int grid_z_size_;
  const float pillar_x_size_;
  const float pillar_y_size_;
  const float pillar_z_size_;
  const float min_x_range_;
  const float min_y_range_;
  const float min_z_range_;
  // end initalizer list

  float *dev_pillar_x_in_coors_;
  float *dev_pillar_y_in_coors_;
  float *dev_pillar_z_in_coors_;
  float *dev_pillar_i_in_coors_;
  int *dev_pillar_count_histo_;

  int *dev_counter_;
  int *dev_pillar_count_;
  float *dev_x_coors_for_sub_;
  float *dev_y_coors_for_sub_;

 public:
  /**
  * @brief Constructor
  * @param[in] max_num_pillars Maximum number of pillars
  * @param[in] max_points_per_pillar Maximum number of points per pillar
  * @param[in] grid_x_size Number of pillars in x-coordinate
  * @param[in] grid_y_size Number of pillars in y-coordinate
  * @param[in] grid_z_size Number of pillars in z-coordinate
  * @param[in] pillar_x_size Size of x-dimension for a pillar
  * @param[in] pillar_y_size Size of y-dimension for a pillar
  * @param[in] pillar_z_size Size of z-dimension for a pillar
  * @param[in] min_x_range Minimum x value for pointcloud
  * @param[in] min_y_range Minimum y value for pointcloud
  * @param[in] min_z_range Minimum z value for pointcloud
  * @param[in] num_box_corners Number of corners for 2D box
  */
  PreProcess(const int max_num_pillars, const int max_points_per_pillar, const int grid_x_size, const int grid_y_size,
             const int grid_z_size, const float pillar_x_size, const float pillar_y_size, const float pillar_z_size,
             const float min_x_range, const float min_y_range, const float min_z_range);
  ~PreProcess();

  /**
  * @brief Preprocessing for input pointcloud
  * @param[in] dev_points Pointcloud array
  * @param[in] in_num_points The number of points
  * @param[in] dev_x_coors X-coordinate indexes for corresponding pillars
  * @param[in] dev_y_coors Y-coordinate indexes for corresponding pillars
  * @param[in] dev_num_points_per_pillar Number of points in corresponding pillars
  * @param[in] dev_pillar_x X-coordinate values for points in each pillar
  * @param[in] dev_pillar_y Y-coordinate values for points in each pillar
  * @param[in] dev_pillar_z Z-coordinate values for points in each pillar
  * @param[in] dev_pillar_i Intensity values for points in each pillar
  * @param[in] dev_x_coors_for_sub_shaped Array for x substraction in the network
  * @param[in] dev_y_coors_for_sub_shaped Array for y substraction in the network
  * @param[in] dev_pillar_feature_mask Mask to make pillars' feature zero where no points in the pillars
  * @param[in] dev_sparse_pillar_map Grid map representation for pillar-occupancy
  * @param[in] host_pillar_count The numnber of valid pillars for an input pointcloud
  * @details Convert pointcloud to pillar representation
  */
  void DoPreProcess(const float *dev_points, const int in_num_points, int *dev_x_coors, int *dev_y_coors,
                    float *dev_num_points_per_pillar, float *dev_pillar_x, float *dev_pillar_y, float *dev_pillar_z,
                    float *dev_pillar_i, float *dev_x_coors_for_sub_shaped, float *dev_y_coors_for_sub_shaped,
                    float *dev_pillar_feature_mask, int *dev_sparse_pillar_map, int *host_pillar_count);
};
}  // namespace pointpillars
