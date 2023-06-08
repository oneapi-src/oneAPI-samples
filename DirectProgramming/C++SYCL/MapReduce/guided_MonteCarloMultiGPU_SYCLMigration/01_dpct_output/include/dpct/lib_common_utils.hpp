//==---- lib_common_utils.hpp ---------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_LIB_COMMON_UTILS_HPP__
#define __DPCT_LIB_COMMON_UTILS_HPP__

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>


namespace dpct {

namespace detail {

template <typename T> inline auto get_memory(T *x) {
  return x;
}

} // namespace detail

} // namespace dpct

#endif // __DPCT_LIB_COMMON_UTILS_HPP__
