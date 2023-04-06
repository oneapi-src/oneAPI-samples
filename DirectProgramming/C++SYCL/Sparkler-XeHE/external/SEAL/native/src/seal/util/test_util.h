// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include "gtest/gtest.h"

namespace intel {
namespace he {

// Generates a vector of type T with size slots small entries
template <typename T>
inline std::vector<T> generateVector(size_t slots, size_t row_size = 0,
                                     size_t n_rows = 2, size_t n_slots = 4) {
  std::vector<T> input(slots, static_cast<T>(0));
  if (row_size == 0) {
    for (size_t i = 0; i < slots; ++i) {
      input[i] = static_cast<T>(i);
    }
  } else {
    for (size_t r = 0; r < n_rows; ++r) {
      for (size_t i = 0; i < n_slots; ++i) {
        input[i + r * row_size] = static_cast<T>(i + r * n_slots);
      }
    }
  }
  return input;
}

template <typename T>
void checkEqual(const std::vector<T>& x, const std::vector<T>& y
                ,T abs_error = T(0.001)) {
  ASSERT_EQ(x.size(), y.size());
  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_NEAR(x[i], y[i],  abs_error);
  }
}

template <typename T>
void checkEqual(const std::vector<std::vector<T>>& x,
                const std::vector<std::vector<T>>& y, T abs_error = T(0.001)) {
  ASSERT_EQ(x.size(), y.size());
  for (size_t i = 0; i < x.size(); ++i) {
    checkEqual(x[i], y[i], abs_error);
  }
}

template <class CollectionT>
double evaluatePolygon_HornerMethod(double input, const CollectionT& coeff) {
  double retval;
  auto it = coeff.rbegin();
  retval = *it;
  for (++it; it != coeff.rend(); ++it) retval = retval * input + *it;
  return retval;
}

template <unsigned int sigmoid_degree>
double approxSigmoid(double x);

template <>
inline double approxSigmoid<3>(double x) {
  // f3(x) ~= 0.5 + 1.20096(x/8) - 0.81562(x/8)^3
  std::array<double, 4> poly = {0.5, 0.15012, 0.0, -0.001593008};
  double retval = evaluatePolygon_HornerMethod(x, poly);
  if (x < -5.0 || retval < 0.0)
    retval = 0.0;
  else if (x > 5.0 || retval > 1.0)
    retval = 1.0;
  return retval;
}

}  // namespace he
}  // namespace intel
