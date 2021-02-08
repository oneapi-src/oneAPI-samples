//==============================================================
// Copyright © 2020-2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once
#include <cmath>
#include <vector>

namespace dnn {

struct ObjectDetection {
  ObjectDetection()
      : x(0.f), y(0.f), z(0.f), length(1.f), width(1.f), height(1.f), yaw(1.f), classId(0), likelihood(1.f) {}

  ObjectDetection(float _x, float _y, float _z, float _l, float _w, float _h, float _yaw, int _classId,
                  float _likelihood)
      : x(_x), y(_y), z(_z), length(_l), width(_w), height(_h), yaw(_yaw), classId(_classId), likelihood(_likelihood) {}

  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  int classId;
  float likelihood;

  std::vector<float> classProbabilities;
};

inline void toDetectionList(const std::vector<float> &detections, std::vector<ObjectDetection> &detectionList) {
  const int NUM_FEATURES = 9;
  std::size_t numberOfDetectedObjects = detections.size() / NUM_FEATURES;
  detectionList.clear();
  detectionList.reserve(numberOfDetectedObjects);

  for (size_t i = 0; i != numberOfDetectedObjects; ++i) {
    detectionList.emplace_back(detections[i * NUM_FEATURES + 0], detections[i * NUM_FEATURES + 1],
                               detections[i * NUM_FEATURES + 2], detections[i * NUM_FEATURES + 3],
                               detections[i * NUM_FEATURES + 4], detections[i * NUM_FEATURES + 5],
                               detections[i * NUM_FEATURES + 6], static_cast<int>(detections[i * NUM_FEATURES + 7]),
                               detections[i * NUM_FEATURES + 8]);
  }
}
}
