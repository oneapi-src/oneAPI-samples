//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "lib_rtl.hpp"

// This C++ model is only used during emulation, so it should functionally
// match the RTL in lib_rtl.v.

SYCL_EXTERNAL extern "C" MyInt54 RtlDSPm27x27u (MyInt27 x, MyInt27 y) {
  return (x * y);
}

