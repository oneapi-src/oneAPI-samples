// =============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
using MyInt27 = ac_int<27, false>;
using MyInt54 = ac_int<54, false>;

SYCL_EXTERNAL extern "C" MyInt54 RtlDSPm27x27u(MyInt27 x, MyInt27 y);

