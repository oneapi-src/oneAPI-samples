//==---- dpct.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_HPP__
#define __DPCT_HPP__

#include <sycl/sycl.hpp>
#include <iostream>
#include <limits.h>
#include <math.h>

#include "device.hpp"
#include "memory.hpp"
#include "util.hpp"

namespace dpct{
enum error_code { success = 0, default_error = 999 };
}

#define DPCT_CHECK_ERROR(expr)                                                 \
  [&]() {                                                                      \
    try {                                                                      \
      expr;                                                                    \
      return dpct::success;                                                    \
    } catch (std::exception const &e) {                                        \
      std::cerr << e.what() << std::endl;                                      \
      return dpct::default_error;                                              \
    }                                                                          \
  }()

#endif // __DPCT_HPP__
