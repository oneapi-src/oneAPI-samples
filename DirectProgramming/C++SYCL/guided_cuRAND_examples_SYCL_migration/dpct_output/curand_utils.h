/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <dpct/rng_utils.hpp>

enum curandRngType {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
    CURAND_RNG_PSEUDO_XORWOW = 101, ///< XORWOW pseudorandom generator
    CURAND_RNG_PSEUDO_MRG32K3A = 121, ///< MRG32k3a pseudorandom generator
    CURAND_RNG_PSEUDO_MTGP32 = 141, ///< Mersenne Twister MTGP32 pseudorandom generator
    CURAND_RNG_PSEUDO_MT19937 = 142, ///< Mersenne Twister MT19937 pseudorandom generator
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161, ///< PHILOX-4x32-10 pseudorandom generator
    CURAND_RNG_QUASI_DEFAULT = 200, ///< Default quasirandom generator
    CURAND_RNG_QUASI_SOBOL32 = 201, ///< Sobol32 quasirandom generator
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,  ///< Scrambled Sobol32 quasirandom generator
    CURAND_RNG_QUASI_SOBOL64 = 203, ///< Sobol64 quasirandom generator
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204  ///< Scrambled Sobol64 quasirandom generator
};

/*
 * CURAND generator types
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum curandRngType curandRngType_t;
/** \endcond */

enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100, ///< Best ordering for pseudorandom results
    CURAND_ORDERING_PSEUDO_DEFAULT = 101, ///< Specific default thread sequence for pseudorandom results, same as CURAND_ORDERING_PSEUDO_BEST
    CURAND_ORDERING_PSEUDO_SEEDED = 102, ///< Specific seeding pattern for fast lower quality pseudorandom results
    CURAND_ORDERING_PSEUDO_LEGACY = 103, ///< Specific legacy sequence for pseudorandom results, guaranteed to remain the same for all cuRAND release
    CURAND_ORDERING_PSEUDO_DYNAMIC = 104, ///< Specific ordering adjusted to the device it is being executed on, provides the best performance
    CURAND_ORDERING_QUASI_DEFAULT = 201 ///< Specific n-dimensional ordering for quasirandom results
};

/*
 * CURAND ordering of results in memory
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum curandOrdering curandOrdering_t;
/** \endcond */



// CUDA API error checking
/*
DPCT1001:0: The statement could not be removed.
*/
/*
DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
*/
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    int err_ = (err);                                                          \
    if (err_ != 0) {                                                           \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  } while (0)

// curand API error checking
#define CURAND_CHECK(err)                                                      \
  do {                                                                         \
    int err_ = (err);                                                          \
    if (err_ != 0) {                                                           \
      std::printf("curand error %d at %s:%d\n", err_, __FILE__, __LINE__);     \
      throw std::runtime_error("curand error");                                \
    }                                                                          \
  } while (0)

template <typename T> void print_vector(const std::vector<T> &data);

template <> void print_vector(const std::vector<float> &data) {
  for (auto &i : data)
    std::printf("%0.6f\n", i);
}

template <> void print_vector(const std::vector<unsigned int> &data) {
  for (auto &i : data)
    std::printf("%d\n", i);
}
