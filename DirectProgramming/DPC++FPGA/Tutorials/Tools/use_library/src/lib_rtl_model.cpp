//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

SYCL_EXTERNAL extern "C" unsigned RtlByteswap(unsigned x) {
  return x << 16 | x >> 16; 
}
