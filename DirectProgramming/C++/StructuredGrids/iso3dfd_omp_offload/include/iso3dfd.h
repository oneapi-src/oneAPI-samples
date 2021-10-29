//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

constexpr float dt = 0.002f;
constexpr float dxyz = 50.0f;
constexpr size_t kHalfLength = 8;
constexpr size_t kMaxTeamSizeLimit = 256;

#define STENCIL_LOOKUP(ir)                                          \
  (coeff[ir] * ((ptr_prev[ix + ir] + ptr_prev[ix - ir]) +           \
                (ptr_prev[ix + ir * n1] + ptr_prev[ix - ir * n1]) + \
                (ptr_prev[ix + ir * dimn1n2] + ptr_prev[ix - ir * dimn1n2])))

#define STENCIL_LOOKUP_Z(ir)                                             \
  (coeff[ir] * (front[ir] + back[ir - 1] + ptr_prev_base[gid + ir] +     \
                ptr_prev_base[gid - ir] + ptr_prev_base[gid + ir * n1] + \
                ptr_prev_base[gid - ir * n1]))

void Usage(const std::string& programName);

void PrintStats(double time, size_t n1, size_t n2, size_t n3,
                size_t num_iterations);

bool WithinEpsilon(float* output, float* reference, size_t dim_x,
                   size_t dim_y, size_t dim_z, size_t radius,
                   const int zadjust, const float delta);

void Initialize(float* ptr_prev, float* ptr_next, float* ptr_vel,
                size_t n1, size_t n2, size_t n3);

bool VerifyResults(float* next_base, float* prev_base, float* vel_base,
                   float* coeff, size_t n1, size_t n2,
                   size_t n3, size_t num_iterations,
                   size_t n1_block, size_t n2_block,
                   size_t n3_block);

bool ValidateInput(size_t n1, size_t n2, size_t n3,
                   size_t n1_block, size_t n2_block,
                   size_t n3_block, size_t num_iterations);
