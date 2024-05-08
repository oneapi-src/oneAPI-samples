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

/* Computation of eigenvalues of a small symmetric, tridiagonal matrix */

// includes, system
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
#include "helper_functions.h"
#include "helper_cuda.h"
#include "config.h"
#include "structs.h"
#include "matlab.h"

// includes, kernels
#include "bisect_kernel_small.dp.hpp"

// includes, file
#include "bisect_small.dp.hpp"

////////////////////////////////////////////////////////////////////////////////
//! Determine eigenvalues for matrices smaller than MAX_SMALL_MATRIX
//! @param TimingIterations  number of iterations for timing
//! @param  input  handles to input data of kernel
//! @param  result handles to result of kernel
//! @param  mat_size  matrix size
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
//! @param  precision  desired precision of eigenvalues
//! @param  iterations  number of iterations for timing
////////////////////////////////////////////////////////////////////////////////
void computeEigenvaluesSmallMatrix(const InputData &input,
                                   ResultDataSmall &result,
                                   const unsigned int mat_size, const float lg,
                                   const float ug, const float precision,
                                   const unsigned int iterations) {
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  for (unsigned int i = 0; i < iterations; ++i) {
    sycl::range<3> blocks(1, 1, 1);
    sycl::range<3> threads(1, 1, MAX_THREADS_BLOCK_SMALL_MATRIX);

    /*
    DPCT1049:16: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:86: 'MAX_THREADS_BLOCK_SMALL_MATRIX' expression was replaced
      with a value. Modify the code to use the original expression, provided
      in comments, if it is correct.
      */
      sycl::local_accessor<float, 1> s_left_acc_ct1(
          sycl::range<1>(512 /*MAX_THREADS_BLOCK_SMALL_MATRIX*/), cgh);
      /*
      DPCT1101:87: 'MAX_THREADS_BLOCK_SMALL_MATRIX' expression was replaced
      with a value. Modify the code to use the original expression, provided
      in comments, if it is correct.
      */
      sycl::local_accessor<float, 1> s_right_acc_ct1(
          sycl::range<1>(512 /*MAX_THREADS_BLOCK_SMALL_MATRIX*/), cgh);
      /*
      DPCT1101:88: 'MAX_THREADS_BLOCK_SMALL_MATRIX' expression was replaced
      with a value. Modify the code to use the original expression, provided
      in comments, if it is correct.
      */
      sycl::local_accessor<unsigned int, 1> s_left_count_acc_ct1(
          sycl::range<1>(512 /*MAX_THREADS_BLOCK_SMALL_MATRIX*/), cgh);
      /*
      DPCT1101:89: 'MAX_THREADS_BLOCK_SMALL_MATRIX' expression was replaced
      with a value. Modify the code to use the original expression, provided
      in comments, if it is correct.
      */
      sycl::local_accessor<unsigned int, 1> s_right_count_acc_ct1(
          sycl::range<1>(512 /*MAX_THREADS_BLOCK_SMALL_MATRIX*/), cgh);
      /*
      DPCT1101:90: 'MAX_THREADS_BLOCK_SMALL_MATRIX + 1' expression was
      replaced with a value. Modify the code to use the original expression,
      provided in comments, if it is correct.
      */
      sycl::local_accessor<unsigned int, 1> s_compaction_list_acc_ct1(
          sycl::range<1>(513 /*MAX_THREADS_BLOCK_SMALL_MATRIX + 1*/), cgh);
      sycl::local_accessor<unsigned int, 0> compact_second_chunk_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> all_threads_converged_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> num_threads_active_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> num_threads_compaction_acc_ct1(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            bisectKernel(
                input.g_a, input.g_b, mat_size, result.g_left, result.g_right,
                result.g_left_count, result.g_right_count, lg, ug, 0, mat_size,
                precision, item_ct1,
                s_left_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                s_right_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                s_left_count_acc_ct1
                    .get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                s_right_count_acc_ct1
                    .get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                s_compaction_list_acc_ct1
                    .get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                compact_second_chunk_acc_ct1, all_threads_converged_acc_ct1,
                num_threads_active_acc_ct1, num_threads_compaction_acc_ct1);
          });
    });
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  sdkStopTimer(&timer);
  getLastCudaError("Kernel launch failed");
  printf("Average time: %f ms (%i iterations)\n",
         sdkGetTimerValue(&timer) / (float)iterations, iterations);

  sdkDeleteTimer(&timer);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for the result for small matrices
//! @param result  handles to the necessary memory
//! @param  mat_size  matrix_size
////////////////////////////////////////////////////////////////////////////////
void initResultSmallMatrix(ResultDataSmall &result,
                           const unsigned int mat_size) {
  result.mat_size_f = sizeof(float) * mat_size;
  result.mat_size_ui = sizeof(unsigned int) * mat_size;

  result.eigenvalues = (float *)malloc(result.mat_size_f);

  // helper variables
  result.zero_f = (float *)malloc(result.mat_size_f);
  result.zero_ui = (unsigned int *)malloc(result.mat_size_ui);

  for (unsigned int i = 0; i < mat_size; ++i) {
    result.zero_f[i] = 0.0f;
    result.zero_ui[i] = 0;

    result.eigenvalues[i] = 0.0f;
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_left = (float *)sycl::malloc_device(
                           result.mat_size_f, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_right = (float *)sycl::malloc_device(
                           result.mat_size_f, dpct::get_in_order_queue())));

  checkCudaErrors(DPCT_CHECK_ERROR(
      result.g_left_count = (unsigned int *)sycl::malloc_device(
          result.mat_size_ui, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      result.g_right_count = (unsigned int *)sycl::malloc_device(
          result.mat_size_ui, dpct::get_in_order_queue())));

  // initialize result memory
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_left, result.zero_f, result.mat_size_f)
          .wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_right, result.zero_f, result.mat_size_f)
          .wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_right_count, result.zero_ui, result.mat_size_ui)
          .wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_left_count, result.zero_ui, result.mat_size_ui)
          .wait()));
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup memory and variables for result for small matrices
//! @param  result  handle to variables
////////////////////////////////////////////////////////////////////////////////
void cleanupResultSmallMatrix(ResultDataSmall &result) {
  freePtr(result.eigenvalues);
  freePtr(result.zero_f);
  freePtr(result.zero_ui);

  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_left, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_right, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_left_count, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_right_count, dpct::get_in_order_queue())));
}

////////////////////////////////////////////////////////////////////////////////
//! Process the result obtained on the device, that is transfer to host and
//! perform basic sanity checking
//! @param  input  handles to input data
//! @param  result  handles to result data
//! @param  mat_size   matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
void processResultSmallMatrix(const InputData &input,
                              const ResultDataSmall &result,
                              const unsigned int mat_size,
                              const char *filename) {
  const unsigned int mat_size_f = sizeof(float) * mat_size;
  const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

  // copy data back to host
  float *left = (float *)malloc(mat_size_f);
  unsigned int *left_count = (unsigned int *)malloc(mat_size_ui);

  checkCudaErrors(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                       .memcpy(left, result.g_left, mat_size_f)
                                       .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(left_count, result.g_left_count, mat_size_ui)
                           .wait()));

  float *eigenvalues = (float *)malloc(mat_size_f);

  for (unsigned int i = 0; i < mat_size; ++i) {
    eigenvalues[left_count[i]] = left[i];
  }

  // save result in matlab format
  writeTridiagSymMatlab(filename, input.a, input.b + 1, eigenvalues, mat_size);

  freePtr(left);
  freePtr(left_count);
  freePtr(eigenvalues);
}
