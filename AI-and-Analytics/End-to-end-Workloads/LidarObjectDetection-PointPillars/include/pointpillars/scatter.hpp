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
 * PointPillar's Scatter.
 * Converts learned features (output from PFE network = 1st CNN) from dense tensors
 * to sparse pseudo image.
 */
class Scatter {
 private:
  const int num_features_;     // The number of features per pillar
  const int max_num_pillars_;  // Maximum number of pillars
  const int grid_x_size_;      // Number of pillars in x-coordinate
  const int grid_y_size_;      // Number of pillars in x-coordinate

 public:
  /**
  * @brief Constructor
  * @param[in] num_features The number of features per pillar
  * @param[in] max_num_pillars Maximum number of pillars
  * @param[in] grid_x_size Number of pillars in x-coordinate
  * @param[in] grid_y_size Number of pillars in y-coordinate
  */
  Scatter(const int num_features, const int max_num_pillars, const int grid_x_size, const int grid_y_size);

  /**
  * @brief Call scatter kernel
  * @param[in] pillar_count The valid number of pillars
  * @param[in] x_coors X-coordinate indexes for corresponding pillars
  * @param[in] y_coors Y-coordinate indexes for corresponding pillars
  * @param[in] pfe_output Output from Pillar Feature Extractor
  * @param[out] scattered_feature Gridmap representation for pillars' feature
  * @details Allocate pillars in gridmap based on index(coordinates) information
  */
  void DoScatter(const int pillar_count, int *x_coors, int *y_coors, float *pfe_output, float *scattered_feature);
};
}  // namespace pointpillars
