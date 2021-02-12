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

namespace dnn {
class Scatter {
 private:
  const int NUM_THREADS_;
  const int MAX_NUM_PILLARS_;
  const int GRID_X_SIZE_;
  const int GRID_Y_SIZE_;

 public:
  /**
  * @brief Constructor
  * @param[in] NUM_THREADS The number of threads to launch kernel
  * @param[in] MAX_NUM_PILLARS Maximum number of pillars
  * @param[in] GRID_X_SIZE Number of pillars in x-coordinate
  * @param[in] GRID_Y_SIZE Number of pillars in y-coordinate
  * @details Captital variables never change after the compile
  */
  Scatter(const int NUM_THREADS, const int MAX_NUM_PILLARS, const int GRID_X_SIZE, const int GRID_Y_SIZE);

  /**
  * @brief Call scatter kernel
  * @param[in] pillar_count The valid number of pillars
  * @param[in] x_coors X-coordinate indexes for corresponding pillars
  * @param[in] y_coors Y-coordinate indexes for corresponding pillars
  * @param[in] pfe_output Output from Pillar Feature Extractor
  * @param[out] scattered_feature Gridmap representation for pillars' feature
  * @details Allocate pillars in gridmap based on index(coordinates) information
  */
  void doScatter(const int pillar_count, int *x_coors, int *y_coors, float *pfe_output, float *scattered_feature);
};
}
