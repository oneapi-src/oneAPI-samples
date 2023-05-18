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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
dpct::constant_memory<float, 1> c_Kernel(KERNEL_LENGTH);

extern "C" void setConvolutionKernel(float *h_Kernel) {
  dpct::get_default_queue()
      .memcpy(c_Kernel.get_ptr(), h_Kernel, KERNEL_LENGTH * sizeof(float))
      .wait();
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define ROWS_HALO_STEPS 1

void convolutionRowsKernel(float *d_Dst, float *d_Src, int imageW,
                                      int imageH, int pitch,
                                      sycl::nd_item<3> item_ct1, float *c_Kernel,
                                      sycl::local_accessor<float, 2> s_Data) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();

  // Offset to the left halo edge
  const int baseX =
      (item_ct1.get_group(2) * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) *
          ROWS_BLOCKDIM_X +
      item_ct1.get_local_id(2);
  const int baseY =
      item_ct1.get_group(1) * ROWS_BLOCKDIM_Y + item_ct1.get_local_id(1);

  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Load main data
#pragma unroll

  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    s_Data[item_ct1.get_local_id(1)]
          [item_ct1.get_local_id(2) + i * ROWS_BLOCKDIM_X] =
              d_Src[i * ROWS_BLOCKDIM_X];
  }

// Load left halo
#pragma unroll

  for (int i = 0; i < ROWS_HALO_STEPS; i++) {
    s_Data[item_ct1.get_local_id(1)]
          [item_ct1.get_local_id(2) + i * ROWS_BLOCKDIM_X] =
              (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
  }

// Load right halo
#pragma unroll

  for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
       i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
    s_Data[item_ct1.get_local_id(1)][item_ct1.get_local_id(2) +
                                     i * ROWS_BLOCKDIM_X] =
        (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
  }

  // Compute and store results
  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
#pragma unroll

  for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
    float sum =0;

#pragma unroll

    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      sum += c_Kernel[KERNEL_RADIUS - j] *
             s_Data[item_ct1.get_local_id(1)]
                   [item_ct1.get_local_id(2) + i * ROWS_BLOCKDIM_X + j];
  }
    d_Dst[i * ROWS_BLOCKDIM_X] = sum;
  }
}

extern "C" void convolutionRowsGPU(float *d_Dst, float *d_Src, int imageW,
                                   int imageH) {
  assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
  assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
  assert(imageH % ROWS_BLOCKDIM_Y == 0);

  sycl::range<3> blocks(1, imageH / ROWS_BLOCKDIM_Y,
                        imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X));
  sycl::range<3> threads(1, ROWS_BLOCKDIM_Y, ROWS_BLOCKDIM_X);

  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    c_Kernel.init();

    auto c_Kernel_ptr_ct1 = c_Kernel.get_ptr();

    sycl::local_accessor<float, 2> s_Data_acc_ct1(sycl::range<2>(
                                                      4 /*4*/,
                                                      160 /*(ROWS_RESULT_STEPS
          + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X*/),
                                                  cgh);

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       convolutionRowsKernel(d_Dst, d_Src, imageW, imageH,
                                             imageW, item_ct1, c_Kernel_ptr_ct1,
                                             s_Data_acc_ct1);
                     });
  });
  getLastCudaError("convolutionRowsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define COLUMNS_BLOCKDIM_X 16
#define COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define COLUMNS_HALO_STEPS 1

void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW,
                                         int imageH, int pitch,
                                         sycl::nd_item<3> item_ct1,
                                         float *c_Kernel,
                                         sycl::local_accessor<float, 2> s_Data) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();

  // Offset to the upper halo edge
  const int baseX =
      item_ct1.get_group(2) * COLUMNS_BLOCKDIM_X + item_ct1.get_local_id(2);
  const int baseY =
      (item_ct1.get_group(1) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) *
          COLUMNS_BLOCKDIM_Y +
      item_ct1.get_local_id(1);
  d_Src += baseY * pitch + baseX;
  d_Dst += baseY * pitch + baseX;

// Main data
#pragma unroll

  for (int i = COLUMNS_HALO_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    s_Data[item_ct1.get_local_id(2)]
          [item_ct1.get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] =
              d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
  }

// Upper halo
#pragma unroll

  for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
    s_Data[item_ct1.get_local_id(2)]
          [item_ct1.get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] =
              (baseY >= -i * COLUMNS_BLOCKDIM_Y)
                  ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch]
                  : 0;
  }

// Lower halo
#pragma unroll

  for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS;
       i++) {
    s_Data[item_ct1.get_local_id(2)]
          [item_ct1.get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] =
              (imageH - baseY > i * COLUMNS_BLOCKDIM_Y)
                  ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch]
                  : 0;
  }

  // Compute and store results
  /*
  DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
#pragma unroll

  for (int i = COLUMNS_HALO_STEPS;
       i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
    float sum =0;
#pragma unroll

    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      sum += c_Kernel[KERNEL_RADIUS - j] *
             s_Data[item_ct1.get_local_id(2)]
                   [item_ct1.get_local_id(1) + i * COLUMNS_BLOCKDIM_Y + j];
    }
    d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
  }
}

extern "C" void convolutionColumnsGPU(float *d_Dst, float *d_Src, int imageW,
                                      int imageH) {
  assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
  assert(imageW % COLUMNS_BLOCKDIM_X == 0);
  assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

  sycl::range<3> blocks(1, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y),
                        imageW / COLUMNS_BLOCKDIM_X);
  sycl::range<3> threads(1, COLUMNS_BLOCKDIM_Y, COLUMNS_BLOCKDIM_X);

  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    c_Kernel.init();

    auto c_Kernel_ptr_ct1 = c_Kernel.get_ptr();

    sycl::local_accessor<float, 2> s_Data_acc_ct1(
        sycl::range<2>(16 /*16*/, 81 /*(COLUMNS_RESULT_STEPS +
     2 * COLUMNS_HALO_STEPS) *
        COLUMNS_BLOCKDIM_Y +
    1*/), cgh);

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       convolutionColumnsKernel(
                           d_Dst, d_Src, imageW, imageH, imageW, item_ct1,
                           c_Kernel_ptr_ct1, s_Data_acc_ct1);
                     });
  });
  getLastCudaError("convolutionColumnsKernel() execution failed\n");
}
