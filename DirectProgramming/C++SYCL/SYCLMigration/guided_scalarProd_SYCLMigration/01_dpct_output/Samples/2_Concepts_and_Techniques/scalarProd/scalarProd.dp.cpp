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

/*
 * This sample calculates scalar products of a
 * given set of input vector pairs
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <helper_functions.h>
#include <helper_cuda.h>

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C" void scalarProdCPU(float *h_C, float *h_A, float *h_B, int vectorN,
                              int elementN);

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
#include "scalarProd_kernel.dp.hpp"
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

// Total number of input vector pairs; arbitrary
const int VECTOR_N = 256;
// Number of elements per vector; arbitrary,
// but strongly preferred to be a multiple of warp size
// to meet memory coalescing constraints
const int ELEMENT_N = 4096;
// Total number of data elements
const int DATA_N = VECTOR_N * ELEMENT_N;

const int DATA_SZ = DATA_N * sizeof(float);
const int RESULT_SZ = VECTOR_N * sizeof(float);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  float *h_A, *h_B, *h_C_CPU, *h_C_GPU;
  float *d_A, *d_B, *d_C;
  double delta, ref, sum_delta, sum_ref, L1norm;
  StopWatchInterface *hTimer = NULL;
  int i;

  printf("%s Starting...\n\n", argv[0]);

  // use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  findCudaDevice(argc, (const char **)argv);

  sdkCreateTimer(&hTimer);

  printf("Initializing data...\n");
  printf("...allocating CPU memory.\n");
  h_A = (float *)malloc(DATA_SZ);
  h_B = (float *)malloc(DATA_SZ);
  h_C_CPU = (float *)malloc(RESULT_SZ);
  h_C_GPU = (float *)malloc(RESULT_SZ);

  printf("...allocating GPU memory.\n");
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_A = (float *)sycl::malloc_device(DATA_SZ, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_B = (float *)sycl::malloc_device(DATA_SZ, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(d_C = (float *)sycl::malloc_device(
                                       RESULT_SZ, dpct::get_in_order_queue())));

  printf("...generating input data in CPU mem.\n");
  srand(123);

  // Generating input data on CPU
  for (i = 0; i < DATA_N; i++) {
    h_A[i] = RandFloat(0.0f, 1.0f);
    h_B[i] = RandFloat(0.0f, 1.0f);
  }

  printf("...copying input data to GPU mem.\n");
  // Copy options data to GPU memory for further processing
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_A, h_A, DATA_SZ).wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_B, h_B, DATA_SZ).wait()));
  printf("Data init done.\n");

  printf("Executing GPU kernel...\n");
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:16: 'ACCUM_N' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> accumResult_acc_ct1(
        sycl::range<1>(1024 /*ACCUM_N*/), cgh);

    int VECTOR_N_ct3 = VECTOR_N;
    int ELEMENT_N_ct4 = ELEMENT_N;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 128) * sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
          scalarProdGPU(
              d_C, d_A, d_B, VECTOR_N_ct3, ELEMENT_N_ct4, item_ct1,
              accumResult_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
  getLastCudaError("scalarProdGPU() execution failed\n");
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  sdkStopTimer(&hTimer);
  printf("GPU time: %f msecs.\n", sdkGetTimerValue(&hTimer));

  printf("Reading back GPU result...\n");
  // Read back GPU results to compare them to CPU results
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_C_GPU, d_C, RESULT_SZ).wait()));

  printf("Checking GPU results...\n");
  printf("..running CPU scalar product calculation\n");
  scalarProdCPU(h_C_CPU, h_A, h_B, VECTOR_N, ELEMENT_N);

  printf("...comparing the results\n");
  // Calculate max absolute difference and L1 distance
  // between CPU and GPU results
  sum_delta = 0;
  sum_ref = 0;

  for (i = 0; i < VECTOR_N; i++) {
    delta = fabs(h_C_GPU[i] - h_C_CPU[i]);
    ref = h_C_CPU[i];
    sum_delta += delta;
    sum_ref += ref;
  }

  L1norm = sum_delta / sum_ref;

  printf("Shutting down...\n");
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_C, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_B, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())));
  free(h_C_GPU);
  free(h_C_CPU);
  free(h_B);
  free(h_A);
  sdkDeleteTimer(&hTimer);

  printf("L1 error: %E\n", L1norm);
  printf((L1norm < 1e-6) ? "Test passed\n" : "Test failed!\n");
  exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}
