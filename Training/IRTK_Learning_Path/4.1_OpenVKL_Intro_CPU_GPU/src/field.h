// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(_MSC_VER) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <cmath>
inline float field(float x, float y, float z)
{
  constexpr float freq = 11.f;
  return (std::sin(freq * x * x * x) + std::sin(freq * y * y) +
          std::cos(freq * z)) /
         3.f;
}
