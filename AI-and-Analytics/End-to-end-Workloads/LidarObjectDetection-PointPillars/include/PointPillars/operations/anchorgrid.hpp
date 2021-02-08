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

#include <cmath>
#include <string>
#include <vector>

#include "PointPillars/PointPillarsConfig.hpp"

namespace dnn {

class AnchorGrid {
 public:
  AnchorGrid(AnchorGridConfig &config);

  ~AnchorGrid();

  AnchorGridConfig mConfig;

  void generateAnchors();

  void moveAnchorsToDevice();

  std::size_t size() { return mNumAnchors; }

  void createAnchorMask(int *devPillarMap, const int pillarMapW, const int pillarMapH, const float pillarSizeX,
                        const float pillarSizeY, int *devAnchorMask, int *devPillarWorkspace);

  // temporary public
  float *mDevAnchorsPx{nullptr};
  float *mDevAnchorsPy{nullptr};
  float *mDevAnchorsPz{nullptr};
  float *mDevAnchorsDx{nullptr};
  float *mDevAnchorsDy{nullptr};
  float *mDevAnchorsDz{nullptr};
  float *mDevAnchorsRo{nullptr};

 private:
  std::size_t mNumAnchors{0u};
  std::size_t mH{0u};
  std::size_t mW{0u};
  std::size_t mC{0u};
  std::size_t mR{0u};

  float *mDevAnchorsRad{nullptr};

  float *mHostAnchorsPx{nullptr};
  float *mHostAnchorsPy{nullptr};
  float *mHostAnchorsPz{nullptr};
  float *mHostAnchorsDx{nullptr};
  float *mHostAnchorsDy{nullptr};
  float *mHostAnchorsDz{nullptr};
  float *mHostAnchorsRo{nullptr};

  float *mHostAnchorsRad;

  void clearHostMemory();
  void clearDeviceMemory();

  void allocateDeviceMemory(const std::size_t &numAnchors);
  void allocateHostMemory(const std::size_t &numAnchors);
};

}  // namespace dnn
