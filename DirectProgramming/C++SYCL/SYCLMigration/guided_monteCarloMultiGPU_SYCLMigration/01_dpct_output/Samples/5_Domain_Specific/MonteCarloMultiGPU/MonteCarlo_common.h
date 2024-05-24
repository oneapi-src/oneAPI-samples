/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MONTECARLO_COMMON_H
#define MONTECARLO_COMMON_H
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "realtype.h"
#include <dpct/rng_utils.hpp>

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
typedef struct dpct_type_495057 {
  float S;
  float X;
  float T;
  float R;
  float V;
} TOptionData;

typedef struct
    // #ifdef __CUDACC__
    //__align__(8)
    // #endif dpct_type_363787
    {
  float Expected;
  float Confidence;
} TOptionValue;

// GPU outputs before CPU postprocessing
typedef struct dpct_type_118506 {
  real Expected;
  real Confidence;
} __TOptionValue;

typedef struct dpct_type_258576 {
  // Device ID for multi-GPU version
  int device;
  // Option count for this plan
  int optionCount;

  // Host-side data source and result destination
  TOptionData *optionData;
  TOptionValue *callValue;

  // Temporary Host-side pinned memory for async + faster data transfers
  __TOptionValue *h_CallValue;

  // Device- and host-side option data
  void *d_OptionData;
  void *h_OptionData;

  // Device-side option values
  void *d_CallValue;

  // Intermediate device-side buffers
  void *d_Buffer;

  // random number generator states
  /*
  DPCT1032:19: A different random number generator is used. You may need to
  adjust the code.
  */
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
      *rngStates;

  // Pseudorandom samples count
  int pathN;

  // Time stamp
  float time;

  int gridSize;
} TOptionPlan;

extern "C" void initMonteCarloGPU(TOptionPlan *plan);
extern "C" void
MonteCarloGPU(TOptionPlan *plan,
              dpct::queue_ptr stream = &dpct::get_in_order_queue());
extern "C" void closeMonteCarloGPU(TOptionPlan *plan);

#endif
