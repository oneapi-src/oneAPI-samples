//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace cl::sycl;

#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>

/*
 * Parameters to define coefficients
 * HALF_LENGTH: Radius of the stencil
 * Sample source code is tested for HALF_LENGTH=8 resulting in
 * 16th order Stencil finite difference kernel
 */
#define DT 0.002f
#define DXYZ 50.0f
#define HALF_LENGTH 8

/*
 * Padding to test and eliminate shared local memory bank conflicts for
 * the shared local memory(slm) version of the kernel executing on GPU
 */
#define PAD 0

bool iso_3dfd_device(cl::sycl::queue&, float*, float*, float*, float*, size_t,
                     size_t, size_t, size_t, size_t, size_t, size_t,
                     unsigned int);

void printTargetInfo(cl::sycl::queue&, unsigned int, unsigned int);

void usage(std::string);

void printStats(double, size_t, size_t, size_t, unsigned int);

bool within_epsilon(float*, float*, const size_t, const size_t, const size_t,
                    const unsigned int, const int, const float);

bool checkGridDimension(size_t, size_t, size_t, unsigned int, unsigned int,
                        unsigned int);

bool checkBlockDimension(cl::sycl::queue&, unsigned int, unsigned int);
