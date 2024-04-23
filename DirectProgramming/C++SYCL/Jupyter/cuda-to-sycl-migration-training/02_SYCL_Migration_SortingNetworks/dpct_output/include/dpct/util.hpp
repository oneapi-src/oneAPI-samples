//==---- util.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_UTIL_HPP__
#define __DPCT_UTIL_HPP__

#include <sycl/sycl.hpp>
#include <complex>
#include <type_traits>
#include <cassert>
#include <cstdint>


namespace dpct {

namespace detail {

template <typename tag, typename T> class generic_error_type {
public:
  generic_error_type() = default;
  generic_error_type(T value) : value{value} {}
  operator T() const { return value; }

private:
  T value;
};

} // namespace detail

using err0 = detail::generic_error_type<struct err0_tag, int>;
using err1 = detail::generic_error_type<struct err1_tag, int>;

} // namespace dpct

#endif // __DPCT_UTIL_HPP__
