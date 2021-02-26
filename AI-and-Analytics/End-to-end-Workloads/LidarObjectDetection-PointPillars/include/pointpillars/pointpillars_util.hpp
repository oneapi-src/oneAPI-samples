//==============================================================
// Copyright © 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once
#include <vector>

namespace pointpillars {

/**
 * 3D-Object representation
 */
struct ObjectDetection {
  ObjectDetection()
      : x(0.f), y(0.f), z(0.f), length(1.f), width(1.f), height(1.f), yaw(1.f), class_id(0), likelihood(1.f) {}

  ObjectDetection(float _x, float _y, float _z, float _l, float _w, float _h, float _yaw, int _class_id,
                  float _likelihood)
      : x(_x),
        y(_y),
        z(_z),
        length(_l),
        width(_w),
        height(_h),
        yaw(_yaw),
        class_id(_class_id),
        likelihood(_likelihood) {}

  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  int class_id;
  float likelihood;

  std::vector<float> class_probabilities;
};

}  // namespace pointpillars
