//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// comparisons.hpp

#ifndef __COMPARISONS_HPP__
#define __COMPARISONS_HPP__

namespace fpga_tools {
template <typename T>
constexpr T Min(T a, T b) {
  return (a < b) ? a : b;
}

template <typename T>
constexpr T Max(T a, T b) {
  return (a > b) ? a : b;
}
}  // namespace fpga_tools
#endif