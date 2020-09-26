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

#if _WIN32 || _WIN64
#define WINDOWS 1
#endif  // _WIN32 || _WIN64

// Random Number Generation (RNG) Distribution parameters
#define ALPHA 0.0f   // Mean
#define SIGMA 0.03f  // Standard Deviation

#if !WINDOWS        // unistd.h not available on windows platforms.
#include <unistd.h> /* getopt() function */
#endif              // !WINDOWS

#include <mkl.h> /* oneMKL, mkl libraries */
#include <CL/sycl.hpp>
#include <dpc_common.hpp>
#include <iomanip> /* setw() function */
#include <iostream>
#include <mkl_rng_sycl.hpp> /* dist() function, mkl namespace */
namespace oneapi {}
using namespace oneapi;
using namespace sycl;
using namespace std;

void ParticleMotion(queue&, const int, float*, float*, float*, float*, size_t*,
                    const size_t, const size_t, const size_t, const size_t,
                    const float);
void CPUParticleMotion(const int, float*, float*, float*, float*, size_t*,
                       const size_t, const size_t, const size_t, unsigned int,
                       const float);
void Usage(const string);
int IsNum(const char*);
bool ValidateDeviceComputation(const size_t*, const size_t*, const size_t,
                               const size_t);
bool CompareMatrices(const size_t*, const size_t*, const size_t);

template <typename T>
void PrintVector(const T*, const size_t);
template <typename T>
void PrintMatrix(const T**, const size_t, const size_t);
template <typename T>
void PrintVectorAsMatrix(T*, const size_t, const size_t);

int parse_cl_args(const int, char* [], size_t*, size_t*, size_t*, size_t*,
                  unsigned int*, unsigned int*);
int parse_cl_args_windows(int, char* [], size_t*, size_t*, size_t*, size_t*,
                          unsigned int*, unsigned int*);
void print_grids(const size_t*, const size_t*, const size_t, const unsigned int,
                 const unsigned int);
void print_validation_results(const size_t*, const size_t*, const size_t,
                              const size_t, const unsigned int,
                              const unsigned int);
void CheckVslError(int);
