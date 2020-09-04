//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;

#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <algorithm>
/*
 * Parameters to define coefficients
 * kHalfLength: Radius of the stencil
 * Sample source code is tested for kHalfLength=8 resulting in
 * 16th order Stencil finite difference kernel
 */
constexpr float dt = 0.002f;
constexpr float dxyz = 50.0f;
constexpr unsigned int kHalfLength = 8;

/*
 * Padding to test and eliminate shared local memory bank conflicts for
 * the shared local memory(slm) version of the kernel executing on GPU
 */
constexpr unsigned int kPad = 0;

bool Iso3dfdDevice(sycl::queue &q, float *ptr_next, float *ptr_prev,
                     float *ptr_vel, float *ptr_coeff, size_t n1, size_t n2,
                     size_t n3, size_t n1_block, size_t n2_block,
                     size_t n3_block, size_t end_z, unsigned int num_iterations);

void PrintTargetInfo(sycl::queue &q, unsigned int dim_x, unsigned int dim_y);

void Usage(const std::string &program_name);

void PrintStats(double time, size_t n1, size_t n2, size_t n3,
                unsigned int num_iterations);

bool WithinEpsilon(float *output, float *reference, const size_t dim_x,
                    const size_t dim_y, const size_t dim_z,
                    const unsigned int radius, const int zadjust,
                    const float delta);

bool CheckGridDimension(size_t n1, size_t n2, size_t n3, unsigned int dim_x,
                        unsigned int dim_y, unsigned int block_z);

bool CheckBlockDimension(sycl::queue &q, unsigned int dim_x, unsigned int dim_y);
