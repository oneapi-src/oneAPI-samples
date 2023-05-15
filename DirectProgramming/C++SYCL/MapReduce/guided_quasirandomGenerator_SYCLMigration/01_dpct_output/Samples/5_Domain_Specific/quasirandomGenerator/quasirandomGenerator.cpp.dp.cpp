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

// CUDA Runtime
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "quasirandomGenerator_common.h"
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// CPU code
////////////////////////////////////////////////////////////////////////////////
extern "C" void initQuasirandomGenerator(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION]);

extern "C" float getQuasirandomValue(
    unsigned int table[QRNG_DIMENSIONS][QRNG_RESOLUTION], int i, int dim);

extern "C" double getQuasirandomValue63(INT64 i, int dim);
extern "C" double MoroInvCNDcpu(unsigned int p);

////////////////////////////////////////////////////////////////////////////////
// GPU code
////////////////////////////////////////////////////////////////////////////////
extern "C" void initTableGPU(
    unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION]);
extern "C" void quasirandomGeneratorGPU(float *d_Output, unsigned int seed,
                                        unsigned int N);
extern "C" void inverseCNDgpu(float *d_Output, unsigned int *d_Input,
                              unsigned int N);

const int N = 1048576;

int main(int argc, char **argv) {
  // Start logs
  printf("%s Starting...\n\n", argv[0]);
  std::cout << "\nRunning on "
            << dpct::get_default_queue().get_device().get_info<sycl::info::device::name>() << "\n";
  unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION];

  float *h_OutputGPU, *d_Output;

  int dim, pos;
  double delta, ref, sumDelta, sumRef, L1norm, gpuTime;

  StopWatchInterface *hTimer = NULL;

  if (sizeof(INT64) != 8) {
    printf("sizeof(INT64) != 8\n");
    return 0;
  }

  sdkCreateTimer(&hTimer);

  printf("Allocating Device memory...\n");
  /*
  DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((d_Output = sycl::malloc_device<float>(
                       QRNG_DIMENSIONS * N, dpct::get_default_queue()),
                   0));

  printf("Allocating CPU memory...\n");
  h_OutputGPU = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));

  printf("Initializing QRNG tables...\n\n");
  initQuasirandomGenerator(tableCPU);

  initTableGPU(tableCPU);

  printf("Testing QRNG...\n\n");
  /*
  DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_default_queue()
                       .memset(d_Output, 0, QRNG_DIMENSIONS * N * sizeof(float))
                       .wait(),
                   0));
  int numIterations = 20;

  for (int i = -1; i < numIterations; i++) {
    if (i == 0) {
      /*
      DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
    }

    quasirandomGeneratorGPU(d_Output, 0, N);
  }

  /*
  DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / (double)numIterations * 1e-3;
  printf(
      "quasirandomGenerator, Throughput = %.4f GNumbers/s, Time = %.5f s, Size "
      "= %u Numbers, NumDevsUsed = %u, Workgroup = %u\n",
      (double)QRNG_DIMENSIONS * (double)N * 1.0E-9 / gpuTime, gpuTime,
      QRNG_DIMENSIONS * N, 1, 128 * QRNG_DIMENSIONS);

  printf("\nReading Device results...\n");
  /*
  DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (dpct::get_default_queue()
           .memcpy(h_OutputGPU, d_Output, QRNG_DIMENSIONS * N * sizeof(float))
           .wait(),
       0));

  printf("Comparing to the CPU results...\n\n");
  sumDelta = 0;
  sumRef = 0;

  for (dim = 0; dim < QRNG_DIMENSIONS; dim++)
    for (pos = 0; pos < N; pos++) {
      ref = getQuasirandomValue63(pos, dim);
      delta = (double)h_OutputGPU[dim * N + pos] - ref;
      sumDelta += fabs(delta);
      sumRef += fabs(ref);
    }

  printf("L1 norm: %E\n", sumDelta / sumRef);

  printf("\nTesting inverseCNDgpu()...\n\n");
  /*
  DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_default_queue()
                       .memset(d_Output, 0, QRNG_DIMENSIONS * N * sizeof(float))
                       .wait(),
                   0));

  for (int i = -1; i < numIterations; i++) {
    if (i == 0) {
      /*
      DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
    }

    inverseCNDgpu(d_Output, NULL, QRNG_DIMENSIONS * N);
  }

  /*
  DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / (double)numIterations * 1e-3;
  printf(
      "quasirandomGenerator-inverse, Throughput = %.4f GNumbers/s, Time = %.5f "
      "s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n",
      (double)QRNG_DIMENSIONS * (double)N * 1E-9 / gpuTime, gpuTime,
      QRNG_DIMENSIONS * N, 1, 128);

  printf("Reading Device results...\n");
  /*
  DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors(
      (dpct::get_default_queue()
           .memcpy(h_OutputGPU, d_Output, QRNG_DIMENSIONS * N * sizeof(float))
           .wait(),
       0));

  printf("\nComparing to the CPU results...\n");
  sumDelta = 0;
  sumRef = 0;
  unsigned int distance = ((unsigned int)-1) / (QRNG_DIMENSIONS * N + 1);

  for (pos = 0; pos < QRNG_DIMENSIONS * N; pos++) {
    unsigned int d = (pos + 1) * distance;
    ref = MoroInvCNDcpu(d);
    delta = (double)h_OutputGPU[pos] - ref;
    sumDelta += fabs(delta);
    sumRef += fabs(ref);
  }

  printf("L1 norm: %E\n\n", L1norm = sumDelta / sumRef);

  printf("Shutting down...\n");
  sdkDeleteTimer(&hTimer);
  free(h_OutputGPU);
  /*
  DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  checkCudaErrors((sycl::free(d_Output, dpct::get_default_queue()), 0));

  exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}
