//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// particle_diffusion: Header file for motionsim.cpp,
// motionsim_kernel.cpp, and utils.cpp
//
// Declares defines, includes, and prototypes needed by the
// Particle Diffusion application.
//

#if defined(WIN32) || defined(WIN64)
#define WINDOWS 1
#else
#define WINDOWS 0
#endif

// RNG Distribution parameters
#define ALPHA 0.0f   // Mean
#define SIGMA 0.03f  // Standard Deviation

#if !defined(WINDOWS)  // unistd.h not available on windows platforms.
#include <unistd.h>    /* getopt() function */
#endif

#include <mkl.h> /* oneMKL, mkl libraries */

#include <CL/sycl.hpp>
#include <dpc_common.hpp>
#include <iomanip> /* setw() function */
#include <iostream>
#include <mkl_rng_sycl.hpp> /* dist() function, mkl namespace */
using namespace sycl;
using namespace std;
#include "motionsim_kernel.cpp"
#include "utils.cpp"

void ParticleMotion(queue&, const size_t, float*, float*, float*, float*,
                    size_t*, const size_t, const size_t, const size_t,
                    const unsigned int, const float, const unsigned int);

void CPUParticleMotion(const size_t, float*, float*, float*, float*, size_t*,
                       const size_t, const size_t, const size_t, unsigned int,
                       const float);
void Usage(string);

bool IsNum(const char*);

template <typename T>
void PrintVectorAsMatrix(T*, size_t, size_t);

template <typename T>
void PrintMatrix(T**, size_t, size_t);

template <typename T>
void PrintVector(T*, size_t);

bool ValidateDeviceComputation(size_t*, size_t*, size_t, const size_t);

bool CompareMatrices(size_t*, size_t*, size_t);
