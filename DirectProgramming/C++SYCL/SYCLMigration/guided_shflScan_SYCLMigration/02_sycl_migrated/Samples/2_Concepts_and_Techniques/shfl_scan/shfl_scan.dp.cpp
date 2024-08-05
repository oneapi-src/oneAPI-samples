//=========================================================
// Modifications Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
//=========================================================

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

// Shuffle intrinsics Sample
// This sample demonstrates the use of the shuffle intrinsic
// First, a simple example of a prefix sum using the shuffle to
// perform a scan operation is provided.
// Secondly, a more involved example of computing an integral image
// using the shuffle intrinsic is provided, where the shuffle
// scan operation and shuffle xor operations are used

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include "shfl_integral_image.dp.hpp"
#include <cmath>

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call,
// then uniformly adding across the input data via the uniform_add<<<>>> kernel.

void shfl_scan_test(int *data, int width, const sycl::nd_item<3> &item_ct1,
                    uint8_t *dpct_local, int *partial_sums = NULL) {
  auto sums = (int *)dpct_local;
  int id = ((item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
            item_ct1.get_local_id(2));
  int lane_id = id % item_ct1.get_sub_group().get_local_range().get(0);
  // determine a warp_id within a block
  int warp_id = item_ct1.get_local_id(2) /
                item_ct1.get_sub_group().get_local_range().get(0);

  // Below is the basic structure of using a shfl instruction
  // for a scan.
  // Record "value" as a variable - we accumulate it along the way
  int value = data[id];

  // Now accumulate in log steps up the chain
  // compute sums, with another thread's value who is
  // distance delta away (i).  Note
  // those threads where the thread 'i' away would have
  // been out of bounds of the warp are unaffected.  This
  // creates the scan sum.

#pragma unroll
  for (int i = 1; i <= width; i *= 2) {
    unsigned int mask = 0xffffffff;
  
    int n = dpct::experimental::shift_sub_group_right(
        mask, item_ct1.get_sub_group(), value, i, width);

    if (lane_id >= i) value += n;
  }

  // value now holds the scan value for the individual thread
  // next sum the largest values for each warp

  // write the sum of the warp to smem
  if (item_ct1.get_local_id(2) %
          item_ct1.get_sub_group().get_local_range().get(0) ==
      item_ct1.get_sub_group().get_local_range().get(0) - 1) {
    sums[warp_id] = value;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  //
  // scan sum the warp sums
  // the same shfl scan operation, but performed on warp sums
  //
  #ifdef NVIDIA_GPU
    if (warp_id == 0) {
   
  #else
    if (warp_id == 0 &&
        lane_id < (item_ct1.get_local_range(2) /
                 item_ct1.get_sub_group().get_local_range().get(0))) {
  #endif
        
    int warp_sum = sums[lane_id];

    int mask = (1 << (item_ct1.get_local_range(2) /
                      item_ct1.get_sub_group().get_local_range().get(0))) -
               1;
    for (int i = 1; i <= (item_ct1.get_local_range(2) /
                          item_ct1.get_sub_group().get_local_range().get(0));
         i *= 2) {
      
      int n = dpct::experimental::shift_sub_group_right(
          mask, item_ct1.get_sub_group(), warp_sum, i,
          (item_ct1.get_local_range(2) /
           item_ct1.get_sub_group().get_local_range().get(0)));

      if (lane_id >= i) warp_sum += n;
    }

    sums[lane_id] = warp_sum;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // perform a uniform add across warps in the block
  // read neighbouring warp's sum and add it to threads value
  int blockSum = 0;

  if (warp_id > 0) {
    blockSum = sums[warp_id - 1];
  }

  value += blockSum;

  // Now write out our result
  data[id] = value;

  // last thread has sum, write write out the block's sum
  if (partial_sums != NULL &&
      item_ct1.get_local_id(2) == item_ct1.get_local_range(2) - 1) {
    partial_sums[item_ct1.get_group(2)] = value;
  }
}

// Uniform add: add partial sums array
void uniform_add(int *data, int *partial_sums, int len,
                 const sycl::nd_item<3> &item_ct1, int &buf) {

  int id = ((item_ct1.get_group(2) * item_ct1.get_local_range(2)) +
            item_ct1.get_local_id(2));

  if (id > len) return;

  if (item_ct1.get_local_id(2) == 0) {
    buf = partial_sums[item_ct1.get_group(2)];
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);
  data[id] += buf;
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor) {
  return ((dividend % divisor) == 0) ? (dividend / divisor)
                                     : (dividend / divisor + 1);
}

// This function verifies the shuffle scan result, for the simple
// prefix sum case.
bool CPUverify(int *h_data, int *h_result, int n_elements) {
  // cpu verify
  for (int i = 0; i < n_elements - 1; i++) {
    h_data[i + 1] = h_data[i] + h_data[i + 1];
  }

  int diff = 0;

  for (int i = 0; i < n_elements; i++) {
    diff += h_data[i] - h_result[i];
  }

  printf("CPU verify result diff (GPUvsCPU) = %d\n", diff);
  bool bTestResult = false;

  if (diff == 0) bTestResult = true;

  StopWatchInterface *hTimer = NULL;
  sdkCreateTimer(&hTimer);
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  for (int j = 0; j < 100; j++)
    for (int i = 0; i < n_elements - 1; i++) {
      h_data[i + 1] = h_data[i] + h_data[i + 1];
    }

  sdkStopTimer(&hTimer);
  double cput = sdkGetTimerValue(&hTimer);
  printf("CPU sum (naive) took %f ms\n", cput / 100);
  return bTestResult;
}

// this verifies the row scan result for synthetic data of all 1's
unsigned int verifyDataRowSums(unsigned int *h_image, int w, int h) {
  unsigned int diff = 0;

  for (int j = 0; j < h; j++) {
    for (int i = 0; i < w; i++) {
      int gold = i + 1;
      diff +=
          abs(static_cast<int>(gold) - static_cast<int>(h_image[j * w + i]));
    }
  }

  return diff;
}

bool shuffle_simple_test(int argc, char **argv) {
  int *h_data, *h_partial_sums, *h_result;
  int *d_data, *d_partial_sums;
  const int n_elements = 65536;
  int sz = sizeof(int) * n_elements;

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

  DPCT_CHECK_ERROR(
      h_data = (int *)sycl::malloc_host(sizeof(int) * n_elements,
                                        q));
  DPCT_CHECK_ERROR(
      h_result = (int *)sycl::malloc_host(sizeof(int) * n_elements,
                                          q));

  // initialize data:
  printf("Computing Simple Sum test\n");
  printf("---------------------------------------------------\n");

  printf("Initialize test data [1, 1, 1...]\n");

  for (int i = 0; i < n_elements; i++) {
    h_data[i] = 1;
  }

  int blockSize = 256;
  int gridSize = n_elements / blockSize;
  int nWarps = blockSize / 32;
  
  int shmem_sz = nWarps * sizeof(int);
  int n_partialSums = n_elements / blockSize;
  int partial_sz = n_partialSums * sizeof(int);

  printf("Scan summation for %d elements, %d partial sums\n", n_elements,
         n_elements / blockSize);

  int p_blockSize = std::min(n_partialSums, blockSize);
  int p_gridSize = iDivUp(n_partialSums, p_blockSize);
  printf("Partial summing %d elements with %d blocks of size %d\n",
         n_partialSums, p_gridSize, p_blockSize);

  // initialize a timer
  dpct::event_ptr start, stop;
  DPCT_CHECK_ERROR(start = new sycl::event());
  DPCT_CHECK_ERROR(stop = new sycl::event());
  float et = 0;
  float inc = 0;

  DPCT_CHECK_ERROR(
      d_data = (int *)sycl::malloc_device(sz, q));
  DPCT_CHECK_ERROR(
      d_partial_sums =
          (int *)sycl::malloc_device(partial_sz, q));
  DPCT_CHECK_ERROR(
      q.memset(d_partial_sums, 0, partial_sz).wait());

  DPCT_CHECK_ERROR(
      h_partial_sums =
          (int *)sycl::malloc_host(partial_sz, q));
  DPCT_CHECK_ERROR(
      q.memcpy(d_data, h_data, sz).wait());

  
      DPCT_CHECK_ERROR(dpct::sync_barrier(start));
 
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(shmem_sz), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *
                              sycl::range<3>(1, 1, blockSize),
                          sycl::range<3>(1, 1, blockSize)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          shfl_scan_test(
              d_data, 32, item_ct1,
              dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              d_partial_sums);
        });
  });

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(shmem_sz), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, p_gridSize) *
                              sycl::range<3>(1, 1, p_blockSize),
                          sycl::range<3>(1, 1, p_blockSize)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          shfl_scan_test(
              d_partial_sums, 32, item_ct1,
              dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get(),
              NULL);
        });
  });
 
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 0> buf_acc_ct1(cgh);

    int *d_data_blockSize_ct0 = d_data + blockSize;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize - 1) *
                                           sycl::range<3>(1, 1, blockSize),
                                       sycl::range<3>(1, 1, blockSize)),
                     [=](sycl::nd_item<3> item_ct1) {
                       uniform_add(d_data_blockSize_ct0, d_partial_sums,
                                   n_elements, item_ct1, buf_acc_ct1);
                     });
  });
 
  
      DPCT_CHECK_ERROR(dpct::sync_barrier(stop));
  DPCT_CHECK_ERROR(stop->wait_and_throw());
  DPCT_CHECK_ERROR(
      inc = (stop->get_profiling_info<
                 sycl::info::event_profiling::command_end>() -
             start->get_profiling_info<
                 sycl::info::event_profiling::command_start>()) /
            1000000.0f);
  et += inc;

  DPCT_CHECK_ERROR(
      q.memcpy(h_result, d_data, sz).wait());
  
      DPCT_CHECK_ERROR(q
                           .memcpy(h_partial_sums, d_partial_sums, partial_sz)
                           .wait());

  printf("Test Sum: %d\n", h_partial_sums[n_partialSums - 1]);
  printf("Time (ms): %f\n", et);
  printf("%d elements scanned in %f ms -> %f MegaElements/s\n", n_elements, et,
         n_elements / (et / 1000.0f) / 1000000.0f);

  bool bTestResult = CPUverify(h_data, h_result, n_elements);

  
  DPCT_CHECK_ERROR(sycl::free(h_data, q));
  
  DPCT_CHECK_ERROR(sycl::free(h_result, q));
  
  DPCT_CHECK_ERROR(sycl::free(h_partial_sums, q));
  
  DPCT_CHECK_ERROR(dpct::dpct_free(d_data, q));
  
  DPCT_CHECK_ERROR(dpct::dpct_free(d_partial_sums, q));

  return bTestResult;
}

// This function tests creation of an integral image using
// synthetic data, of size 1920x1080 pixels greyscale.
bool shuffle_integral_image_test() {
  char *d_data;
  unsigned int *h_image;
  unsigned int *d_integral_image;
  int w = 1920;
  int h = 1080;
  int n_elements = w * h;
  int sz = sizeof(unsigned int) * n_elements;

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

  printf("\nComputing Integral Image Test on size %d x %d synthetic data\n", w,
         h);
  printf("---------------------------------------------------\n");
  DPCT_CHECK_ERROR(h_image = (unsigned int *)sycl::malloc_host(
                                       sz, q));
  // fill test "image" with synthetic 1's data
  memset(h_image, 0, sz);

  // each thread handles 16 values, use 1 block/row
  int blockSize = iDivUp(w, 16);
  // launch 1 block / row
  int gridSize = h;

  // Create a synthetic image for testing
  DPCT_CHECK_ERROR(
      d_data = (char *)sycl::malloc_device(sz, q));
  DPCT_CHECK_ERROR(
      d_integral_image = (unsigned int *)sycl::malloc_device(
          n_elements * sizeof(int) * 4, q));
  DPCT_CHECK_ERROR(
      q.memset(d_data, 1, sz).wait());
  DPCT_CHECK_ERROR(
      q.memset(d_integral_image, 0, sz).wait());

  dpct::event_ptr start, stop;
  start = new sycl::event();
  stop = new sycl::event();
  float et = 0;
  unsigned int err;

  // Execute scan line prefix sum kernel, and time it
  dpct::sync_barrier(start);
 
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<unsigned int, 1> sums_acc_ct1(sycl::range<1>(128),
                                                       cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *
                              sycl::range<3>(1, 1, blockSize),
                          sycl::range<3>(1, 1, blockSize)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          shfl_intimage_rows(
              reinterpret_cast<sycl::uint4 *>(d_data),
              reinterpret_cast<sycl::uint4 *>(d_integral_image), item_ct1,
              sums_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });
  dpct::sync_barrier(stop);
  DPCT_CHECK_ERROR(stop->wait_and_throw());
  DPCT_CHECK_ERROR(
      et = (stop->get_profiling_info<
                sycl::info::event_profiling::command_end>() -
            start->get_profiling_info<
                sycl::info::event_profiling::command_start>()) /
           1000000.0f);
  printf("Method: Fast  Time (GPU Timer): %f ms ", et);

  // verify the scan line results
  DPCT_CHECK_ERROR(
      q.memcpy(h_image, d_integral_image, sz).wait());
  err = verifyDataRowSums(h_image, w, h);
  printf("Diff = %d\n", err);

  // Execute column prefix sum kernel and time it
  sycl::range<3> blockSz(1, 8, 32);
  sycl::range<3> testGrid(1, 1, w / blockSz[2]);

  dpct::sync_barrier(start);
  
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<unsigned int, 2> sums_acc_ct1(sycl::range<2>(32, 9),
                                                       cgh);

    cgh.parallel_for(sycl::nd_range<3>(testGrid * blockSz, blockSz),
                     [=](sycl::nd_item<3> item_ct1)
                         [[intel::reqd_sub_group_size(32)]] {
                           shfl_vertical_shfl((unsigned int *)d_integral_image,
                                              w, h, item_ct1, sums_acc_ct1);
                         });
  });
  dpct::sync_barrier(stop);
  DPCT_CHECK_ERROR(stop->wait_and_throw());
  DPCT_CHECK_ERROR(
      et = (stop->get_profiling_info<
                sycl::info::event_profiling::command_end>() -
            start->get_profiling_info<
                sycl::info::event_profiling::command_start>()) /
           1000000.0f);
  printf("Method: Vertical Scan  Time (GPU Timer): %f ms ", et);

  // Verify the column results
  DPCT_CHECK_ERROR(
      q.memcpy(h_image, d_integral_image, sz).wait());
  printf("\n");

  int finalSum = h_image[w * h - 1];
  printf("CheckSum: %d, (expect %dx%d=%d)\n", finalSum, w, h, w * h);

  
      DPCT_CHECK_ERROR(dpct::dpct_free(d_data, q));
  DPCT_CHECK_ERROR(
      dpct::dpct_free(d_integral_image, q));
  
      DPCT_CHECK_ERROR(sycl::free(h_image, q));
  // verify final sum: if the final value in the corner is the same as the size
  // of the buffer (all 1's) then the integral image was generated successfully
  return (finalSum == w * h) ? true : false;
}

int main(int argc, char *argv[]) {

  int dpct_device = 0;
  printf("Starting shfl_scan\n");

  dpct::device_info deviceProp;
  DPCT_CHECK_ERROR(
      dpct_device = dpct::dev_mgr::instance().current_device_id());

  DPCT_CHECK_ERROR(dpct::get_device_info(
      deviceProp, dpct::dev_mgr::instance().get_device(dpct_device)));

  std::cout << "Running on device: "<< deviceProp.get_name() << "\n";
  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
         deviceProp.get_major_version(), deviceProp.get_minor_version(),
         deviceProp.get_max_compute_units());


  bool bTestResult = true;
  bool simpleTest = shuffle_simple_test(argc, argv);
  bool intTest = shuffle_integral_image_test();

  bTestResult = simpleTest & intTest;

  exit((bTestResult) ? EXIT_SUCCESS : EXIT_FAILURE);
}
