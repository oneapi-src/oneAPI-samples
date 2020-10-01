//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// motionsim.hpp: Header file for motionsim.cpp,
// motionsim_kernel.cpp, and utils.cpp
//
// Constant expressions, includes, and prototypes needed by the application.
//

// Random Number Generation (RNG) Distribution parameters
constexpr float alpha = 0.0f;   // Mean
constexpr float sigma = 0.03f;  // Standard Deviation

#if _WIN32 || _WIN64
#define WINDOWS 1
#endif  // _WIN32 || _WIN64
// unistd.h not available on windows platforms
#if !WINDOWS
#include <unistd.h>
#endif  // !WINDOWS

#include <CL/sycl.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"
// For backwards compatibility with MKL-Beta09
#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#include "oneapi/rng.hpp"
#else  // __has_include("oneapi/mkl.hpp")
#include <mkl.h>
#include "mkl_sycl.hpp"
#endif  // __has_include("oneapi/mkl.hpp")

void ParticleMotion(sycl::queue&, const int, float*, float*, float*, float*,
                    size_t*, const size_t, const size_t, const size_t,
                    const size_t, const float);
void CPUParticleMotion(const int, float*, float*, float*, float*, size_t*,
                       const size_t, const size_t, const size_t, unsigned int,
                       const float);
void Usage();
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

int ParseArgs(const int, char* [], size_t*, size_t*, size_t*, int*,
              unsigned int*, unsigned int*);
int ParseArgsWindows(int, char* [], size_t*, size_t*, size_t*, int*,
                     unsigned int*, unsigned int*);
void PrintGrids(const size_t*, const size_t*, const size_t, const unsigned int,
                const unsigned int);
void PrintValidationResults(const size_t*, const size_t*, const size_t,
                            const size_t, const unsigned int,
                            const unsigned int);
void CheckVslError(int);
