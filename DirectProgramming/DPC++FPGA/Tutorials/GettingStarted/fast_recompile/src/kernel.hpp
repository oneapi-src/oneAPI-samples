//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

// tolerance used in floating point comparisons
constexpr float kTol = 0.001;

// array size of vectors a, b and c
constexpr size_t kArraySize = 32;

void RunKernel(std::vector<float> &vec_a, std::vector<float> &vec_b,
               std::vector<float> &vec_r);
