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
 * This is a simple test showing huge access speed gap
 * between aligned and misaligned structures
 * (those having/missing __align__ keyword).
 * It measures per-element copy throughput for
 * aligned and misaligned structures on
 * big chunks of data.
 */

// includes, system
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <helper_cuda.h>  // helper functions for CUDA error checking and initialization
#include <helper_functions.h>  // helper utility functions

////////////////////////////////////////////////////////////////////////////////
// Misaligned types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned char uint8;

typedef unsigned short int uint16;

typedef struct dpct_type_135292 {
  unsigned char r, g, b, a;
} RGBA8_misaligned;

typedef struct dpct_type_670789 {
  unsigned int l, a;
} LA32_misaligned;

typedef struct dpct_type_317807 {
  unsigned int r, g, b;
} RGB32_misaligned;

typedef struct dpct_type_426295 {
  unsigned int r, g, b, a;
} RGBA32_misaligned;

////////////////////////////////////////////////////////////////////////////////
// Aligned types
////////////////////////////////////////////////////////////////////////////////
typedef struct __dpct_align__(4) dpct_type_157154 {
  unsigned char r, g, b, a;
}
RGBA8;

typedef unsigned int I32;

typedef struct __dpct_align__(8) dpct_type_102693 {
  unsigned int l, a;
}
LA32;

typedef struct __dpct_align__(16) dpct_type_278528 {
  unsigned int r, g, b;
}
RGB32;

typedef struct __dpct_align__(16) dpct_type_726353 {
  unsigned int r, g, b, a;
}
RGBA32;

////////////////////////////////////////////////////////////////////////////////
// Because G80 class hardware natively supports global memory operations
// only with data elements of 4, 8 and 16 bytes, if structure size
// exceeds 16 bytes, it can't be efficiently read or written,
// since more than one global memory non-coalescable load/store instructions
// will be generated, even if __align__ option is supplied.
// "Structure of arrays" storage strategy offers best performance
// in general case. See section 5.1.2 of the Programming Guide.
////////////////////////////////////////////////////////////////////////////////
typedef struct __dpct_align__(16) dpct_type_111555 {
  RGBA32 c1, c2;
}
RGBA32_2;

////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
// Round a / b to nearest higher integer value
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// Round a / b to nearest lower integer value
int iDivDown(int a, int b) { return a / b; }

// Align a to nearest higher multiple of b
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

// Align a to nearest lower multiple of b
int iAlignDown(int a, int b) { return a - a % b; }

////////////////////////////////////////////////////////////////////////////////
// Simple CUDA kernel.
// Copy is carried out on per-element basis,
// so it's not per-byte in case of padded structures.
////////////////////////////////////////////////////////////////////////////////
template <class TData>
void testKernel(TData *d_odata, TData *d_idata, int numElements,
                const sycl::nd_item<3> &item_ct1) {
  const int tid = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
  const int numThreads =
      item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

  for (int pos = tid; pos < numElements; pos += numThreads) {
    d_odata[pos] = d_idata[pos];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Validation routine for simple copy kernel.
// We must know "packed" size of TData (number_of_fields * sizeof(simple_type))
// and compare only these "packed" parts of the structure,
// containing actual user data. The compiler behavior with padding bytes
// is undefined, since padding is merely a placeholder
// and doesn't contain any user data.
////////////////////////////////////////////////////////////////////////////////
template <class TData>
int testCPU(TData *h_odata, TData *h_idata, int numElements,
            int packedElementSize) {
  for (int pos = 0; pos < numElements; pos++) {
    TData src = h_idata[pos];
    TData dst = h_odata[pos];

    for (int i = 0; i < packedElementSize; i++)
      if (((char *)&src)[i] != ((char *)&dst)[i]) {
        return 0;
      }
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
// Memory chunk size in bytes. Reused for test
const int MEM_SIZE = 50000000;
const int NUM_ITERATIONS = 32;

// GPU input and output data
unsigned char *d_idata, *d_odata;
// CPU input data and instance of GPU output data
unsigned char *h_idataCPU, *h_odataGPU;
StopWatchInterface *hTimer = NULL;

template <class TData>
int runTest(int packedElementSize, int memory_size) {
  const int totalMemSizeAligned = iAlignDown(memory_size, sizeof(TData));
  const int numElements = iDivDown(memory_size, sizeof(TData));

  // Clean output buffer before current test
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memset(d_odata, 0, memory_size).wait()));
  // Run test
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      TData *d_odata_ct0 = (TData *)d_odata;
      TData *d_idata_ct1 = (TData *)d_idata;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 64) *
                                             sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                         testKernel<TData>(d_odata_ct0, d_idata_ct1,
                                           numElements, item_ct1);
                       });
    });
    getLastCudaError("testKernel() execution failed\n");
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  sdkStopTimer(&hTimer);
  double gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;
  printf("Avg. time: %f ms / Copy throughput: %f GB/s.\n", gpuTime,
         (double)totalMemSizeAligned / (gpuTime * 0.001 * 1073741824.0));

  // Read back GPU results and run validation
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                       .memcpy(h_odataGPU, d_odata, memory_size)
                                       .wait()));
  int flag = testCPU((TData *)h_odataGPU, (TData *)h_idataCPU, numElements,
                     packedElementSize);

  printf(flag ? "\tTEST OK\n" : "\tTEST FAILURE\n");

  return !flag;
}

int main(int argc, char **argv) {
  int i, nTotalFailures = 0;

  int devID;
  dpct::device_info deviceProp;
  printf("[%s] - Starting...\n", argv[0]);

  // find first CUDA device
  devID = 0;// findCudaDevice(argc, (const char **)argv);

  // get number of SMs on this GPU
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_device_info(
      deviceProp, dpct::dev_mgr::instance().get_device(devID))));
  printf("[%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n",
         deviceProp.get_name(), deviceProp.get_max_compute_units(),
         /*
         DPCT1005:14: The SYCL device version is different from CUDA Compute
         Compatibility. You may need to rewrite this code.
         */
         _ConvertSMVer2Cores(deviceProp.get_major_version(),
                             deviceProp.get_minor_version()),
         /*
         DPCT1005:15: The SYCL device version is different from CUDA Compute
         Compatibility. You may need to rewrite this code.
         */
         _ConvertSMVer2Cores(deviceProp.get_major_version(),
                             deviceProp.get_minor_version()) *
             deviceProp.get_max_compute_units());

  // Anything that is less than 192 Cores will have a scaled down workload
  float scale_factor =
      /*
      DPCT1005:16: The SYCL device version is different from CUDA Compute
      Compatibility. You may need to rewrite this code.
      */
      std::max((192.0f / (_ConvertSMVer2Cores(deviceProp.get_major_version(),
                                              deviceProp.get_minor_version()) *
                          (float)deviceProp.get_max_compute_units())),
               1.0f);

  int MemorySize = (int)(MEM_SIZE / scale_factor) &
                   0xffffff00;  // force multiple of 256 bytes

  printf("> Compute scaling value = %4.2f\n", scale_factor);
  printf("> Memory Size = %d\n", MemorySize);

  sdkCreateTimer(&hTimer);

  printf("Allocating memory...\n");
  h_idataCPU = (unsigned char *)malloc(MemorySize);
  h_odataGPU = (unsigned char *)malloc(MemorySize);
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_idata = (unsigned char *)sycl::malloc_device(
                           MemorySize, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_odata = (unsigned char *)sycl::malloc_device(
                           MemorySize, dpct::get_in_order_queue())));

  printf("Generating host input data array...\n");

  for (i = 0; i < MemorySize; i++) {
    h_idataCPU[i] = (i & 0xFF) + 1;
  }

  printf("Uploading input data to GPU memory...\n");
  checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                       .memcpy(d_idata, h_idataCPU, MemorySize)
                                       .wait()));

  printf("Testing misaligned types...\n");
  printf("uint8...\n");
  nTotalFailures += runTest<uint8>(1, MemorySize);

  printf("uint16...\n");
  nTotalFailures += runTest<uint16>(2, MemorySize);

  printf("RGBA8_misaligned...\n");
  nTotalFailures += runTest<RGBA8_misaligned>(4, MemorySize);

  printf("LA32_misaligned...\n");
  nTotalFailures += runTest<LA32_misaligned>(8, MemorySize);

  printf("RGB32_misaligned...\n");
  nTotalFailures += runTest<RGB32_misaligned>(12, MemorySize);

  printf("RGBA32_misaligned...\n");
  nTotalFailures += runTest<RGBA32_misaligned>(16, MemorySize);

  printf("Testing aligned types...\n");
  printf("RGBA8...\n");
  nTotalFailures += runTest<RGBA8>(4, MemorySize);

  printf("I32...\n");
  nTotalFailures += runTest<I32>(4, MemorySize);

  printf("LA32...\n");
  nTotalFailures += runTest<LA32>(8, MemorySize);

  printf("RGB32...\n");
  nTotalFailures += runTest<RGB32>(12, MemorySize);

  printf("RGBA32...\n");
  nTotalFailures += runTest<RGBA32>(16, MemorySize);

  printf("RGBA32_2...\n");
  nTotalFailures += runTest<RGBA32_2>(32, MemorySize);

  printf("\n[alignedTypes] -> Test Results: %d Failures\n", nTotalFailures);

  printf("Shutting down...\n");
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_idata, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_odata, dpct::get_in_order_queue())));
  free(h_odataGPU);
  free(h_idataCPU);

  sdkDeleteTimer(&hTimer);

  if (nTotalFailures != 0) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
