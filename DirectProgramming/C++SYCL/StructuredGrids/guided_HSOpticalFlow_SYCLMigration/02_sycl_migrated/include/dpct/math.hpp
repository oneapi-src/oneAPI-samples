//==---- math.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_MATH_HPP__
#define __DPCT_MATH_HPP__

#include <sycl/sycl.hpp>

namespace dpct {

// min function overloads.
// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
inline double min(const double a, const float b) {
  return sycl::fmin(a, static_cast<double>(b));
}
inline double min(const float a, const double b) {
  return sycl::fmin(static_cast<double>(a), b);
}
inline float min(const float a, const float b) { return sycl::fmin(a, b); }
inline double min(const double a, const double b) { return sycl::fmin(a, b); }
inline std::uint32_t min(const std::uint32_t a, const std::int32_t b) {
  return sycl::min(a, static_cast<std::uint32_t>(b));
}
inline std::uint32_t min(const std::int32_t a, const std::uint32_t b) {
  return sycl::min(static_cast<std::uint32_t>(a), b);
}
inline std::int32_t min(const std::int32_t a, const std::int32_t b) {
  return sycl::min(a, b);
}
inline std::uint32_t min(const std::uint32_t a, const std::uint32_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::int64_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::int64_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
inline std::int64_t min(const std::int64_t a, const std::int64_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::uint64_t b) {
  return sycl::min(a, b);
}
inline std::uint64_t min(const std::uint64_t a, const std::int32_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::int32_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
inline std::uint64_t min(const std::uint64_t a, const std::uint32_t b) {
  return sycl::min(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t min(const std::uint32_t a, const std::uint64_t b) {
  return sycl::min(static_cast<std::uint64_t>(a), b);
}
// max function overloads.
// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
inline double max(const double a, const float b) {
  return sycl::fmax(a, static_cast<double>(b));
}
inline double max(const float a, const double b) {
  return sycl::fmax(static_cast<double>(a), b);
}
inline float max(const float a, const float b) { return sycl::fmax(a, b); }
inline double max(const double a, const double b) { return sycl::fmax(a, b); }
inline std::uint32_t max(const std::uint32_t a, const std::int32_t b) {
  return sycl::max(a, static_cast<std::uint32_t>(b));
}
inline std::uint32_t max(const std::int32_t a, const std::uint32_t b) {
  return sycl::max(static_cast<std::uint32_t>(a), b);
}
inline std::int32_t max(const std::int32_t a, const std::int32_t b) {
  return sycl::max(a, b);
}
inline std::uint32_t max(const std::uint32_t a, const std::uint32_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::int64_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::int64_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}
inline std::int64_t max(const std::int64_t a, const std::int64_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::uint64_t b) {
  return sycl::max(a, b);
}
inline std::uint64_t max(const std::uint64_t a, const std::int32_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::int32_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}
inline std::uint64_t max(const std::uint64_t a, const std::uint32_t b) {
  return sycl::max(a, static_cast<std::uint64_t>(b));
}
inline std::uint64_t max(const std::uint32_t a, const std::uint64_t b) {
  return sycl::max(static_cast<std::uint64_t>(a), b);
}

} // namespace dpct

#endif // __DPCT_MATH_HPP__
