// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "field.h"

#include <vector>
#include <cstddef>

// The structured regular volume expects a data buffer that contains
// all voxel values. We create it here by sampling the field() function, but
// this data could be the output of a simulation, or loaded from disk.
inline std::vector<float> createVoxels(size_t res)
{
  std::vector<float> voxels(res * res * res);
  for (size_t z = 0; z < res; ++z)
    for (size_t y = 0; y < res; ++y)
      for (size_t x = 0; x < res; ++x) {
        const float fx   = x / static_cast<float>(res);
        const float fy   = y / static_cast<float>(res);
        const float fz   = z / static_cast<float>(res);
        const size_t idx = z * res * res + y * res + x;
        voxels[idx]      = field(fx, fy, fz);
      }
  return voxels;
}
