//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _GAMMA_UTILS_IMGALGORITHM_HPP
#define _GAMMA_UTILS_IMGALGORITHM_HPP

#include <cmath>
#include <cstdint>

using namespace std;

// struct to store fractal that image will fill from
class ImgFractal {
 private:
  const int32_t _width;
  const int32_t _height;

  double _cx = -0.7436;
  double _cy = 0.1319;

  double _magn = 2000000.0;
  int _maxIterations = 1000;

 public:
  ImgFractal(int32_t width, int32_t height) : _width(width), _height(height) {}

  double operator()(int32_t x, int32_t y) const {
    double fx = (double(x) - double(_width) / 2) * (1 / _magn) + _cx;
    double fy = (double(y) - double(_height) / 2) * (1 / _magn) + _cy;

    double res = 0;
    double nx = 0;
    double ny = 0;
    double val = 0;

    for (int i = 0; nx * nx + ny * ny <= 4 && i < _maxIterations; ++i) {
      val = nx * nx - ny * ny + fx;
      ny = 2 * nx * ny + fy;
      nx = val;

      res += exp(-sqrt(nx * nx + ny * ny));
    }

    return res;
  }
};

#endif  // _GAMMA_UTILS_IMGALGORITHM_HPP
