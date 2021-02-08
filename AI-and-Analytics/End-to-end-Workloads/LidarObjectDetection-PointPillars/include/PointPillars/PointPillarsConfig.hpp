//==============================================================
// Copyright © 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace dnn {

struct Anchor {
  Anchor(float _x, float _y, float _z) : x(_x), y(_y), z(_z), dz(0.){};
  Anchor(float _x, float _y, float _z, float _dz) : x(_x), y(_y), z(_z), dz(_dz){};

  float x{1.f};
  float y{1.f};
  float z{1.f};
  float dz{0.f};
};

struct AnchorGridConfig {
  float minXRange{0.0f};  // defines the area covered
  float maxXRange{1.0f};  // defines the area covered
  float minYRange{0.0f};  // defines the area covered
  float maxYRange{1.0f};  // defines the area covered
  float minZRange{0.0f};  // defines the area covered
  float maxZRange{1.0f};  // defines the area covered

  float xStride{0.01f};
  float yStride{0.01f};

  std::vector<Anchor> anchors = {Anchor(1.0f, 2.0f, 1.5f)};
  std::vector<float> rotations = {0.f, M_PI_2};
};

struct PointPillarsConfig {
  std::string pfeModelFile{"pfe"};
  std::string rpnModelFile{"rpn"};
  float minXRange{0.f};      // defines the area covered by the algorithm
  float maxXRange{69.12f};   // defines the area covered by the algorithm
  float minYRange{-39.68f};  // defines the area covered by the algorithm
  float maxYRange{39.68f};   // defines the area covered by the algorithm
  float minZRange{-3.0f};    // defines the area covered by the algorithm
  float maxZRange{1.0f};     // defines the area covered by the algorithm
  float rpnScale{0.5f};      // upsample_scale / downsample scale
  float pillarXSize{0.16f};  // pillar voxelization size along X
  float pillarYSize{0.16f};  // pillar voxelization size along Y
  float pillarZSize{4.0f};   // pillar voxelization size along Z
  float xStride{0.32f};
  float yStride{0.32f};
  std::size_t maxNumPillars{12000};
  std::size_t numClasses{1};
  std::vector<Anchor> anchors = {Anchor(1.6f, 3.9f, 1.56f)};
  std::vector<std::string> classes = {"Car"};
  std::size_t maxNumPointsPerPillar{100};
  std::size_t pillarFeatures{64};
  std::size_t gridXSize{432};  // (maxXRange - minXRange) / pillarXSize
  std::size_t gridYSize{496};  // (maxYRange - minYRange) / pillarYSize
  std::size_t gridZSize{1};    // (maxZRange - minZRange) / pillarZSize
};
}
