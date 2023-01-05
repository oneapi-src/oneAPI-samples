//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef _GAMMA_UTILS_IMGPIXEL_HPP
#define _GAMMA_UTILS_IMGPIXEL_HPP

#include <cstdint>
#include <ostream>

using namespace std;

// struct to store a pixel of image
struct ImgPixel {
  uint8_t b;
  uint8_t g;
  uint8_t r;
  uint8_t a;

  bool operator==(ImgPixel const& other) const {
    return (b == other.b) && (g == other.g) && (r == other.r) && (a == other.a);
  }

  bool operator!=(ImgPixel const& other) const { return !(*this == other); }

  void set(uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha) {
    b = blue;
    g = green;
    r = red;
    a = alpha;
  }
};

ostream& operator<<(ostream& output, ImgPixel const& pixel) {
  return output << "(" << unsigned(pixel.r) << ", " << unsigned(pixel.g) << ", "
                << unsigned(pixel.b) << ", " << unsigned(pixel.a) << ")";
}

#endif  // _GAMMA_UTILS_IMGPIXEL_HPP
