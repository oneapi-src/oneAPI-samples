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
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/

// CUDA Runtime
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include "histogram_common.h"

const int numRuns = 16;
const static char *sSDKsample = "[histogram]\0";
sycl::queue sycl_queue;
int main(int argc, char **argv) {
  uchar *h_Data;
  uint *h_HistogramCPU, *h_HistogramGPU;
  uchar *d_Data;
  uint *d_Histogram;
  StopWatchInterface *hTimer = NULL;
  int PassFailFlag = 1;
  uint byteCount = 64 * 1048576;
  uint uiSizeMult = 1;

  dpct::device_info deviceProp;
  deviceProp.set_major_version(0);
  deviceProp.set_minor_version(0);

  // set logfile name and start logs
  printf("[%s] - Starting...\n", sSDKsample);

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  int dev = 0;
  
  DPCT_CHECK_ERROR(dpct::get_device_info(
      deviceProp, dpct::dev_mgr::instance().get_device(dev)));

  printf("SYCL device [%s] has %d Multi-Processors, Compute %d.%d\n",
         deviceProp.get_name(), deviceProp.get_max_compute_units(),
         deviceProp.get_major_version(),
         deviceProp.get_minor_version());

  sdkCreateTimer(&hTimer);

  // Optional Command-line multiplier to increase size of array to histogram
  if (checkCmdLineFlag(argc, (const char **)argv, "sizemult")) {
    uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    uiSizeMult = MAX(1, MIN(uiSizeMult, 10));
    byteCount *= uiSizeMult;
  }

  printf("Initializing data...\n");
  printf("...allocating CPU memory.\n");
  h_Data = (uchar *)malloc(byteCount);
  h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
  h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

  printf("...generating input data\n");
  srand(2009);

  for (uint i = 0; i < byteCount; i++) {
    h_Data[i] = rand() % 256;
  }

  printf("...allocating GPU memory and copying input data\n\n");
  DPCT_CHECK_ERROR(d_Data = (uchar *)sycl::malloc_device(
                                       byteCount, sycl_queue));
  DPCT_CHECK_ERROR(
      d_Histogram = sycl::malloc_device<uint>(HISTOGRAM256_BIN_COUNT,
                                              sycl_queue));
  DPCT_CHECK_ERROR(
      sycl_queue.memcpy(d_Data, h_Data, byteCount).wait());

  {
    printf("Starting up 64-bin histogram...\n\n");
    initHistogram64(sycl_queue);

    printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n",
           byteCount, numRuns);

    for (int iter = -1; iter < numRuns; iter++) {
      // iter == -1 -- warmup iteration
      if (iter == 0) {
        //dpct::get_current_device().queues_wait_and_throw();
        sycl_queue.wait();
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);
      }

      histogram64(d_Histogram, d_Data, byteCount, sycl_queue);
    }

    //dpct::get_current_device().queues_wait_and_throw();
    sycl_queue.wait();
    sdkStopTimer(&hTimer);
    double dAvgSecs =
        1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
    printf("histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs,
           ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf(
        "histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, "
        "NumDevsUsed = %u, Workgroup = %u\n",
        (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1,
        HISTOGRAM64_THREADBLOCK_SIZE);

    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
    
    DPCT_CHECK_ERROR(sycl_queue
                             .memcpy(h_HistogramGPU, d_Histogram,
                                     HISTOGRAM64_BIN_COUNT * sizeof(uint))
                             .wait());

    printf(" ...histogram64CPU()\n");
    histogram64CPU(h_HistogramCPU, h_Data, byteCount);

    printf(" ...comparing the results...\n");

    for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
      if (h_HistogramGPU[i] != h_HistogramCPU[i]) {
        PassFailFlag = 0;
      }

    printf(PassFailFlag ? " ...64-bin histograms match\n\n"
                        : " ***64-bin histograms do not match!!!***\n\n");

    printf("Shutting down 64-bin histogram...\n\n\n");
    closeHistogram64(sycl_queue);
  }

  {
    printf("Initializing 256-bin histogram...\n");
    initHistogram256(sycl_queue);

    printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n",
           byteCount, numRuns);

    for (int iter = -1; iter < numRuns; iter++) {
      // iter == -1 -- warmup iteration
      if (iter == 0) {
        DPCT_CHECK_ERROR(sycl_queue.wait());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);
      }

      histogram256(d_Histogram, d_Data, byteCount, sycl_queue);
    }

    sycl_queue.wait();
    sdkStopTimer(&hTimer);
    double dAvgSecs =
        1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
    printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n",
           dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf(
        "histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, "
        "NumDevsUsed = %u, Workgroup = %u\n",
        (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1,
        HISTOGRAM256_THREADBLOCK_SIZE);

    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
    
    DPCT_CHECK_ERROR(sycl_queue
                             .memcpy(h_HistogramGPU, d_Histogram,
                                     HISTOGRAM256_BIN_COUNT * sizeof(uint))
                             .wait());

    printf(" ...histogram256CPU()\n");
    histogram256CPU(h_HistogramCPU, h_Data, byteCount);

    printf(" ...comparing the results\n");

    for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
      if (h_HistogramGPU[i] != h_HistogramCPU[i]) {
        PassFailFlag = 0;
      }

    printf(PassFailFlag ? " ...256-bin histograms match\n\n"
                        : " ***256-bin histograms do not match!!!***\n\n");

    printf("Shutting down 256-bin histogram...\n\n\n");
    closeHistogram256(sycl_queue);
  
  }
  printf("Shutting down...\n");
  sdkDeleteTimer(&hTimer);
  
  DPCT_CHECK_ERROR(sycl::free(d_Histogram, sycl_queue));
  DPCT_CHECK_ERROR(sycl::free(d_Data, sycl_queue));
  free(h_HistogramGPU);
  free(h_HistogramCPU);
  free(h_Data);

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  printf("%s - Test Summary\n", sSDKsample);
  // pass or fail (for both 64 bit and 256 bit histograms)
  if (!PassFailFlag) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
