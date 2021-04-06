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
#include <cmath>
#include <string>
#include <vector>

#include "pointpillars/pointpillars_config.hpp"

namespace pointpillars {

/**
 * AnchorGrid
 *
 * The AnchorGrid class generates anchors in different sizes and orientations for every location in the grid.
 * Anchor based methods are used in object detection in which a list of predefined boxes are refined by a CNN.
 *
 */
class AnchorGrid {
 public:
  /**
  * @brief Constructor
  *
  * Class used to generate the anchor grid which is used as a prior box list during object detection
  *
  * @param[in] config Configuration used to generate anchors
  */
  AnchorGrid(AnchorGridConfig &config);
  ~AnchorGrid();

  AnchorGridConfig config_;

  // Pointers to device memory locations for the anchors
  float *dev_anchors_px_{nullptr};
  float *dev_anchors_py_{nullptr};
  float *dev_anchors_pz_{nullptr};
  float *dev_anchors_dx_{nullptr};
  float *dev_anchors_dy_{nullptr};
  float *dev_anchors_dz_{nullptr};
  float *dev_anchors_ro_{nullptr};

  // Get size/number of anchors
  std::size_t size() { return num_anchors_; }

  // Generate default anchors
  void GenerateAnchors();

  // Creates an anchor mask that can be used to ignore anchors in regions without points
  // Input is the current pillar map (map, width, height, size in x, size in y, size in z)
  // Output are the created anchors
  void CreateAnchorMask(int *dev_pillar_map, const int pillar_map_w, const int pillar_map_h, const float pillar_size_x,
                        const float pillar_size_y, int *dev_anchor_mask, int *dev_pillar_workspace);

 private:
  std::size_t num_anchors_{0u};
  std::size_t mh_{0u};
  std::size_t mw_{0u};
  std::size_t mc_{0u};
  std::size_t mr_{0u};

  // Anchor pointers on the host
  // Only required for initialization
  float *dev_anchors_rad_{nullptr};
  float *host_anchors_px_{nullptr};
  float *host_anchors_py_{nullptr};
  float *host_anchors_pz_{nullptr};
  float *host_anchors_dx_{nullptr};
  float *host_anchors_dy_{nullptr};
  float *host_anchors_dz_{nullptr};
  float *host_anchors_ro_{nullptr};
  float *host_anchors_rad_;

  // Clear host memory
  void ClearHostMemory();

  // Clear device memory
  void ClearDeviceMemory();

  // Allocate host memory
  void AllocateHostMemory();

  // Allocate device memory
  void AllocateDeviceMemory();

  // Move anchors from the host system to the target execution device
  void MoveAnchorsToDevice();

  // Internal function to create anchor mask
  void MaskAnchors(const float *dev_anchors_px, const float *dev_anchors_py, const int *dev_pillar_map,
                   int *dev_anchor_mask, const float *dev_anchors_rad, const float min_x_range, const float min_y_range,
                   const float pillar_x_size, const float pillar_y_size, const int grid_x_size, const int grid_y_size,
                   const int c, const int r, const int h, const int w);
};

}  // namespace pointpillars
