//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

using namespace sycl;

void RunKernel(queue& q, buffer<float,1>& buf_a, buffer<float,1>& buf_b,
               buffer<float,1>& buf_r, size_t size);
