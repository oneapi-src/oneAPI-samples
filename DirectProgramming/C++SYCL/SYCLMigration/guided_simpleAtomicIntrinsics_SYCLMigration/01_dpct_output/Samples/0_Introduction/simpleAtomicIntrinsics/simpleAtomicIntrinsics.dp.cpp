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

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

// includes, system
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// Includes CUDA

// Utilities and timing functions
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>  // helper functions for CUDA error check

// Includes, kernels
#include "simpleAtomicIntrinsics_kernel.dp.hpp"

const char *sampleName = "simpleAtomicIntrinsics";

////////////////////////////////////////////////////////////////////////////////
// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" bool computeGold(int *gpuData, const int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s starting...\n", sampleName);

  runTest(argc, argv);

  printf("%s completed, returned %s\n", sampleName,
         testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  dpct::queue_ptr stream;
  // This will pick the best possible CUDA capable device
  findCudaDevice(argc, (const char **)argv);

  StopWatchInterface *timer;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 11;
  unsigned int memSize = sizeof(int) * numData;

  // allocate mem for the result on host side
  int *hOData;
  checkCudaErrors(DPCT_CHECK_ERROR(
      hOData = (int *)sycl::malloc_host(memSize, dpct::get_in_order_queue())));

  // initialize the memory
  for (unsigned int i = 0; i < numData; i++) hOData[i] = 0;

  // To make the AND and XOR tests generate something other than 0...
  hOData[8] = hOData[10] = 0xff;

  /*
  DPCT1025:15: The SYCL queue is created ignoring the flag and priority options.
  */
  checkCudaErrors(
      DPCT_CHECK_ERROR(stream = dpct::get_current_device().create_queue()));
  // allocate device memory for result
  int *dOData;
  checkCudaErrors(DPCT_CHECK_ERROR(dOData = (int *)sycl::malloc_device(
                                       memSize, dpct::get_in_order_queue())));
  // copy host memory to device to initialize to zero
  checkCudaErrors(DPCT_CHECK_ERROR(stream->memcpy(dOData, hOData, memSize)));

  // execute the kernel
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                                             sycl::range<3>(1, 1, numThreads),
                                         sycl::range<3>(1, 1, numThreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         testKernel(dOData, item_ct1);
                       });

  // Copy result from device to host
  checkCudaErrors(DPCT_CHECK_ERROR(stream->memcpy(hOData, dOData, memSize)));
  checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));

  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  // Compute reference solution
  testResult = computeGold(hOData, numThreads * numBlocks);

  // Cleanup memory
  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(hOData, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(dOData, dpct::get_in_order_queue())));
}
