//  Copyright (c) 2023 Intel Corporation
//  SPDX-License-Identifier: MIT

// comparisons.hpp

#ifndef __COMPARISONS_HPP__
#define __COMPARISONS_HPP__

template <typename T>
constexpr T min(T a, T b) {
    return (a < b) ? a : b;
}

template <typename T>
constexpr T max(T a, T b) {
    return (a > b) ? a : b;
}

#endif