//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>

constexpr float dt = 0.002f;
constexpr float dxyz = 50.0f;
constexpr unsigned int kHalfLength = 8;
constexpr unsigned int kMaxTeamSizeLimit = 256;

#define STENCIL_LOOKUP(ir)                                          \
  (coeff[ir] * ((ptr_prev[ix + ir] + ptr_prev[ix - ir]) +           \
                (ptr_prev[ix + ir * n1] + ptr_prev[ix - ir * n1]) + \
                (ptr_prev[ix + ir * dimn1n2] + ptr_prev[ix - ir * dimn1n2])))

#define STENCIL_LOOKUP_Z(ir)                                             \
  (coeff[ir] * (front[ir] + back[ir - 1] + ptr_prev_base[gid + ir] +     \
                ptr_prev_base[gid - ir] + ptr_prev_base[gid + ir * n1] + \
                ptr_prev_base[gid - ir * n1]))

void Usage(const std::string& programName);

void PrintStats(double time, unsigned int n1, unsigned int n2, unsigned int n3,
                unsigned int num_iterations);

bool WithinEpsilon(float* output, float* reference, unsigned int dim_x,
                   unsigned int dim_y, unsigned int dim_z, unsigned int radius,
                   const int zadjust, const float delta);

void Initialize(float* ptr_prev, float* ptr_next, float* ptr_vel,
                unsigned int n1, unsigned int n2, unsigned int n3);

bool VerifyResults(float* next_base, float* prev_base, float* vel_base,
                   float* coeff, unsigned int n1, unsigned int n2,
                   unsigned int n3, unsigned int num_iterations,
                   unsigned int n1_block, unsigned int n2_block,
                   unsigned int n3_block);

bool ValidateInput(unsigned int n1, unsigned int n2, unsigned int n3,
                   unsigned int n1_block, unsigned int n2_block,
                   unsigned int n3_block, unsigned int num_iterations);
