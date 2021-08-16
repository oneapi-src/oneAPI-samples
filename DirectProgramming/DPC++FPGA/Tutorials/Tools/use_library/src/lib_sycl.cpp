//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

SYCL_EXTERNAL float SyclSquare(float x) { return x * x; }
SYCL_EXTERNAL float SyclSqrt(float x) { return sqrt(x); }
