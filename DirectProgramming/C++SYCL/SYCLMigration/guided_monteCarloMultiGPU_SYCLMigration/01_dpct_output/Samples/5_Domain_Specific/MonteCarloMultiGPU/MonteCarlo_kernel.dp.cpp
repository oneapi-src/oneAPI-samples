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

////////////////////////////////////////////////////////////////////////////////
// Global types
////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <dpct/rng_utils.hpp>

#include "MonteCarlo_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper reduction template
// Please see the "reduction" CUDA Sample for more information
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_reduction.dp.hpp"
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side data structures
////////////////////////////////////////////////////////////////////////////////
#define MAX_OPTIONS (1024 * 1024)

// Preprocessed input option data
typedef struct dpct_type_164939 {
  real S;
  real X;
  real MuByT;
  real VBySqrtT;
} __TOptionData;

////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut payoff functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
inline float endCallValue(float S, float X, float r, float MuByT,
                                     float VBySqrtT) {
  float callValue = S * sycl::native::exp(MuByT + VBySqrtT * r) - X;
  return (callValue > 0.0F) ? callValue : 0.0F;
}

inline double endCallValue(double S, double X, double r,
                                      double MuByT, double VBySqrtT) {
  double callValue = S * sycl::exp(MuByT + VBySqrtT * r) - X;
  return (callValue > 0.0) ? callValue : 0.0;
}

#define THREAD_N 256

////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy.
////////////////////////////////////////////////////////////////////////////////
/*
DPCT1110:3: The total declared local variable size in device function
MonteCarloOneBlockPerOption exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void MonteCarloOneBlockPerOption(
    /*
    DPCT1032:31: A different random number generator is used. You may need to
    adjust the code.
    */
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *__restrict rngStates,
    const __TOptionData *__restrict d_OptionData,
    __TOptionValue *__restrict d_CallValue, int pathN, int optionN,
    const sycl::nd_item<3> &item_ct1, real *s_SumCall, real *s_Sum2Call) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
  sycl::sub_group tile32 = item_ct1.get_sub_group();

  const int SUM_N = THREAD_N;

  // determine global thread id
  int tid = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);

  // Copy random number state to local memory for efficiency
  /*
  DPCT1032:32: A different random number generator is used. You may need to
  adjust the code.
  */
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
      localState = rngStates[tid];
  for (int optionIndex = item_ct1.get_group(2); optionIndex < optionN;
       optionIndex += item_ct1.get_group_range(2)) {
    const real S = d_OptionData[optionIndex].S;
    const real X = d_OptionData[optionIndex].X;
    const real MuByT = d_OptionData[optionIndex].MuByT;
    const real VBySqrtT = d_OptionData[optionIndex].VBySqrtT;

    // Cycle through the entire samples array:
    // derive end stock price for each path
    // accumulate partial integrals into intermediate shared memory buffer
    for (int iSum = item_ct1.get_local_id(2); iSum < SUM_N;
         iSum += item_ct1.get_local_range(2)) {
      __TOptionValue sumCall = {0, 0};

#pragma unroll 8
      for (int i = iSum; i < pathN; i += SUM_N) {
        real r =
            localState.generate<oneapi::mkl::rng::device::gaussian<float>, 1>();
        real callValue = endCallValue(S, X, r, MuByT, VBySqrtT);
        sumCall.Expected += callValue;
        sumCall.Confidence += callValue * callValue;
      }

      s_SumCall[iSum] = sumCall.Expected;
      s_Sum2Call[iSum] = sumCall.Confidence;
    }

    // Reduce shared memory accumulators
    // and write final result to global memory
    /*
    DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    sumReduce<real, SUM_N, THREAD_N>(s_SumCall, s_Sum2Call, cta, tile32,
                                     &d_CallValue[optionIndex], item_ct1);
  }
}

/*
DPCT1032:33: A different random number generator is used. You may need to adjust
the code.
*/
static void rngSetupStates(
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *rngState,
    int device_id, const sycl::nd_item<3> &item_ct1) {
  // determine global thread id
  int tid = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
  // Each threadblock gets different seed,
  // Threads within a threadblock get different sequence numbers
  /*
  DPCT1105:34: The mcg59 random number generator is used. The subsequence
  argument "item_ct1.get_local_id(2)" is ignored. You need to verify the
  migration.
  */
  rngState[tid] =
      dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>(
          item_ct1.get_group(2) + item_ct1.get_group_range(2) * device_id, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////

extern "C" void initMonteCarloGPU(TOptionPlan *plan) {
  checkCudaErrors(
      DPCT_CHECK_ERROR(plan->d_OptionData = (void *)sycl::malloc_device(
                           sizeof(__TOptionData) * (plan->optionCount),
                           dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(plan->d_CallValue = (void *)sycl::malloc_device(
                           sizeof(__TOptionValue) * (plan->optionCount),
                           dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      plan->h_OptionData =
          (void *)sycl::malloc_host(sizeof(__TOptionData) * (plan->optionCount),
                                    dpct::get_in_order_queue())));
  // Allocate internal device memory
  checkCudaErrors(
      DPCT_CHECK_ERROR(plan->h_CallValue = sycl::malloc_host<__TOptionValue>(
                           (plan->optionCount), dpct::get_in_order_queue())));
  // Allocate states for pseudo random number generators
  checkCudaErrors(DPCT_CHECK_ERROR(
      plan->rngStates = sycl::malloc_device<
          dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>>(
          plan->gridSize * THREAD_N, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memset(plan->rngStates, 0,
                  /*
                  DPCT1032:36: A different random number generator is used. You
                  may need to adjust the code.
                  */
                  plan->gridSize * THREAD_N *
                      sizeof(dpct::rng::device::rng_generator<
                             oneapi::mkl::rng::device::mcg59<1>>))
          .wait()));

  // place each device pathN random numbers apart on the random number sequence
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *plan_rngStates_ct0 = plan->rngStates;
    int plan_device_ct1 = plan->device;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, plan->gridSize) *
                                           sycl::range<3>(1, 1, THREAD_N),
                                       sycl::range<3>(1, 1, THREAD_N)),
                     [=](sycl::nd_item<3> item_ct1) {
                       rngSetupStates(plan_rngStates_ct0, plan_device_ct1,
                                      item_ct1);
                     });
  });
  getLastCudaError("rngSetupStates kernel failed.\n");
}

// Compute statistics and deallocate internal device memory
extern "C" void closeMonteCarloGPU(TOptionPlan *plan) {
  for (int i = 0; i < plan->optionCount; i++) {
    const double RT = plan->optionData[i].R * plan->optionData[i].T;
    const double sum = plan->h_CallValue[i].Expected;
    const double sum2 = plan->h_CallValue[i].Confidence;
    const double pathN = plan->pathN;
    // Derive average from the total sum and discount by riskfree rate
    plan->callValue[i].Expected = (float)(exp(-RT) * sum / pathN);
    // Standard deviation
    double stdDev = sqrt((pathN * sum2 - sum * sum) / (pathN * (pathN - 1)));
    // Confidence width; in 95% of all cases theoretical value lies within these
    // borders
    plan->callValue[i].Confidence =
        (float)(exp(-RT) * 1.96 * stdDev / sqrt(pathN));
  }

  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(plan->rngStates, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      sycl::free(plan->h_CallValue, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      sycl::free(plan->h_OptionData, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(plan->d_CallValue, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(plan->d_OptionData, dpct::get_in_order_queue())));
}

// Main computations
extern "C" void MonteCarloGPU(TOptionPlan *plan, dpct::queue_ptr stream) {
  __TOptionValue *h_CallValue = plan->h_CallValue;

  if (plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS) {
    printf("MonteCarloGPU(): bad option count.\n");
    return;
  }

  __TOptionData *h_OptionData = (__TOptionData *)plan->h_OptionData;

  for (int i = 0; i < plan->optionCount; i++) {
    const double T = plan->optionData[i].T;
    const double R = plan->optionData[i].R;
    const double V = plan->optionData[i].V;
    const double MuByT = (R - 0.5 * V * V) * T;
    const double VBySqrtT = V * sqrt(T);
    h_OptionData[i].S = (real)plan->optionData[i].S;
    h_OptionData[i].X = (real)plan->optionData[i].X;
    h_OptionData[i].MuByT = (real)MuByT;
    h_OptionData[i].VBySqrtT = (real)VBySqrtT;
  }

  checkCudaErrors(DPCT_CHECK_ERROR(
      stream->memcpy(plan->d_OptionData, h_OptionData,
                     plan->optionCount * sizeof(__TOptionData))));

  stream->submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:37: 'SUM_N' expression was replaced with a value. Modify the code
    to use the original expression, provided in comments, if it is correct.
    */
    sycl::local_accessor<real, 1> s_SumCall_acc_ct1(
        sycl::range<1>(256 /*SUM_N*/), cgh);
    /*
    DPCT1101:38: 'SUM_N' expression was replaced with a value. Modify the code
    to use the original expression, provided in comments, if it is correct.
    */
    sycl::local_accessor<real, 1> s_Sum2Call_acc_ct1(
        sycl::range<1>(256 /*SUM_N*/), cgh);

    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>
        *plan_rngStates_ct0 = plan->rngStates;
    const __TOptionData *plan_d_OptionData_ct1 =
        (__TOptionData *)(plan->d_OptionData);
    __TOptionValue *plan_d_CallValue_ct2 =
        (__TOptionValue *)(plan->d_CallValue);
    int plan_pathN_ct3 = plan->pathN;
    int plan_optionCount_ct4 = plan->optionCount;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, plan->gridSize) *
                              sycl::range<3>(1, 1, THREAD_N),
                          sycl::range<3>(1, 1, THREAD_N)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          MonteCarloOneBlockPerOption(
              plan_rngStates_ct0, plan_d_OptionData_ct1, plan_d_CallValue_ct2,
              plan_pathN_ct3, plan_optionCount_ct4, item_ct1,
              s_SumCall_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              s_Sum2Call_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
  getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");

  checkCudaErrors(DPCT_CHECK_ERROR(
      stream->memcpy(h_CallValue, plan->d_CallValue,
                     plan->optionCount * sizeof(__TOptionValue))));

  // cudaDeviceSynchronize();
}
