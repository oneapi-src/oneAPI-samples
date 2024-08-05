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

/* Computation of eigenvalues of a large symmetric, tridiagonal matrix */

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
#include "util.h"
#include "matlab.h"

#include "bisect_large.dp.hpp"

// includes, kernels
#include "bisect_kernel_large.dp.hpp"
#include "bisect_kernel_large_onei.dp.hpp"
#include "bisect_kernel_large_multi.dp.hpp"

////////////////////////////////////////////////////////////////////////////////
//! Initialize variables and memory for result
//! @param  result handles to memory
//! @param  matrix_size  size of the matrix
////////////////////////////////////////////////////////////////////////////////
void initResultDataLargeMatrix(ResultDataLarge &result,
                               const unsigned int mat_size) {
  // helper variables to initialize memory
  unsigned int zero = 0;
  unsigned int mat_size_f = sizeof(float) * mat_size;
  unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;

  float *tempf = (float *)malloc(mat_size_f);
  unsigned int *tempui = (unsigned int *)malloc(mat_size_ui);

  for (unsigned int i = 0; i < mat_size; ++i) {
    tempf[i] = 0.0f;
    tempui[i] = 0;
  }

  // number of intervals containing only one eigenvalue after the first step
  checkCudaErrors(DPCT_CHECK_ERROR(
      result.g_num_one =
          sycl::malloc_device<unsigned int>(1, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_num_one, &zero, sizeof(unsigned int))
          .wait()));

  // number of (thread) blocks of intervals with multiple eigenvalues after
  // the first iteration
  checkCudaErrors(DPCT_CHECK_ERROR(
      result.g_num_blocks_mult =
          sycl::malloc_device<unsigned int>(1, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_num_blocks_mult, &zero, sizeof(unsigned int))
          .wait()));

  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_left_one = (float *)sycl::malloc_device(
                           mat_size_f, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_right_one = (float *)sycl::malloc_device(
                           mat_size_f, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_pos_one = (unsigned int *)sycl::malloc_device(
                           mat_size_ui, dpct::get_in_order_queue())));

  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_left_mult = (float *)sycl::malloc_device(
                           mat_size_f, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_right_mult = (float *)sycl::malloc_device(
                           mat_size_f, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      result.g_left_count_mult = (unsigned int *)sycl::malloc_device(
          mat_size_ui, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      result.g_right_count_mult = (unsigned int *)sycl::malloc_device(
          mat_size_ui, dpct::get_in_order_queue())));

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(result.g_left_one, tempf, mat_size_f)
                           .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(result.g_right_one, tempf, mat_size_f)
                           .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(result.g_pos_one, tempui, mat_size_ui)
                           .wait()));

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(result.g_left_mult, tempf, mat_size_f)
                           .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(result.g_right_mult, tempf, mat_size_f)
                           .wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_left_count_mult, tempui, mat_size_ui)
          .wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_right_count_mult, tempui, mat_size_ui)
          .wait()));

  checkCudaErrors(DPCT_CHECK_ERROR(
      result.g_blocks_mult = (unsigned int *)sycl::malloc_device(
          mat_size_ui, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(result.g_blocks_mult, tempui, mat_size_ui)
                           .wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      result.g_blocks_mult_sum = (unsigned int *)sycl::malloc_device(
          mat_size_ui, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(result.g_blocks_mult_sum, tempui, mat_size_ui)
          .wait()));

  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_lambda_mult = (float *)sycl::malloc_device(
                           mat_size_f, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(result.g_lambda_mult, tempf, mat_size_f)
                           .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(result.g_pos_mult = (unsigned int *)sycl::malloc_device(
                           mat_size_ui, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(result.g_pos_mult, tempf, mat_size_ui)
                           .wait()));
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup result memory
//! @param result  handles to memory
////////////////////////////////////////////////////////////////////////////////
void cleanupResultDataLargeMatrix(ResultDataLarge &result) {
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_num_one, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_num_blocks_mult, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_left_one, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_right_one, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_pos_one, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_left_mult, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_right_mult, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_left_count_mult, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_right_count_mult, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_blocks_mult, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_blocks_mult_sum, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_lambda_mult, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::dpct_free(result.g_pos_mult, dpct::get_in_order_queue())));
}

////////////////////////////////////////////////////////////////////////////////
//! Run the kernels to compute the eigenvalues for large matrices
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  precision  desired precision of eigenvalues
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
//! @param  iterations  number of iterations (for timing)
////////////////////////////////////////////////////////////////////////////////
void computeEigenvaluesLargeMatrix(const InputData &input,
                                   const ResultDataLarge &result,
                                   const unsigned int mat_size,
                                   const float precision, const float lg,
                                   const float ug,
                                   const unsigned int iterations) {
  sycl::range<3> blocks(1, 1, 1);
  sycl::range<3> threads(1, 1, MAX_THREADS_BLOCK);

  StopWatchInterface *timer_step1 = NULL;
  StopWatchInterface *timer_step2_one = NULL;
  StopWatchInterface *timer_step2_mult = NULL;
  StopWatchInterface *timer_total = NULL;
  sdkCreateTimer(&timer_step1);
  sdkCreateTimer(&timer_step2_one);
  sdkCreateTimer(&timer_step2_mult);
  sdkCreateTimer(&timer_total);

  sdkStartTimer(&timer_total);

  // do for multiple iterations to improve timing accuracy
  for (unsigned int iter = 0; iter < iterations; ++iter) {
    sdkStartTimer(&timer_step1);
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:73: '2 * MAX_THREADS_BLOCK + 1' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<float, 1> s_left_acc_ct1(
          sycl::range<1>(513 /*2 * MAX_THREADS_BLOCK + 1*/), cgh);
      /*
      DPCT1101:74: '2 * MAX_THREADS_BLOCK + 1' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<float, 1> s_right_acc_ct1(
          sycl::range<1>(513 /*2 * MAX_THREADS_BLOCK + 1*/), cgh);
      /*
      DPCT1101:75: '2 * MAX_THREADS_BLOCK + 1' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<unsigned short, 1> s_left_count_acc_ct1(
          sycl::range<1>(513 /*2 * MAX_THREADS_BLOCK + 1*/), cgh);
      /*
      DPCT1101:76: '2 * MAX_THREADS_BLOCK + 1' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<unsigned short, 1> s_right_count_acc_ct1(
          sycl::range<1>(513 /*2 * MAX_THREADS_BLOCK + 1*/), cgh);
      /*
      DPCT1101:77: '2 * MAX_THREADS_BLOCK + 1' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<unsigned short, 1> s_compaction_list_acc_ct1(
          sycl::range<1>(513 /*2 * MAX_THREADS_BLOCK + 1*/), cgh);
      sycl::local_accessor<unsigned int, 0> compact_second_chunk_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> all_threads_converged_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> num_threads_active_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> num_threads_compaction_acc_ct1(cgh);
      /*
      DPCT1101:78: '2 * MAX_THREADS_BLOCK + 1' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<unsigned short, 1> s_cl_helper_acc_ct1(
          sycl::range<1>(513 /*2 * MAX_THREADS_BLOCK + 1*/), cgh);
      sycl::local_accessor<unsigned int, 0> num_blocks_mult_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> num_mult_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> offset_mult_lambda_acc_ct1(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(blocks * threads, threads),
          [=](sycl::nd_item<3> item_ct1) {
            bisectKernelLarge(
                input.g_a, input.g_b, mat_size, lg, ug, 0, mat_size, precision,
                result.g_num_one, result.g_num_blocks_mult, result.g_left_one,
                result.g_right_one, result.g_pos_one, result.g_left_mult,
                result.g_right_mult, result.g_left_count_mult,
                result.g_right_count_mult, result.g_blocks_mult,
                result.g_blocks_mult_sum, item_ct1,
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
                num_threads_active_acc_ct1, num_threads_compaction_acc_ct1,
                s_cl_helper_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                num_blocks_mult_acc_ct1, num_mult_acc_ct1,
                offset_mult_lambda_acc_ct1);
          });
    });

    getLastCudaError("Kernel launch failed.");
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    sdkStopTimer(&timer_step1);

    // get the number of intervals containing one eigenvalue after the first
    // processing step
    unsigned int num_one_intervals;
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(&num_one_intervals, result.g_num_one, sizeof(unsigned int))
            .wait()));

    sycl::range<3> grid_onei(1, 1, 1);
    grid_onei[2] = getNumBlocksLinear(num_one_intervals, MAX_THREADS_BLOCK);
    sycl::range<3> threads_onei(1, 1, 1);
    // use always max number of available threads to better balance load times
    // for matrix data
    threads_onei[2] = MAX_THREADS_BLOCK;

    // compute eigenvalues for intervals that contained only one eigenvalue
    // after the first processing step
    sdkStartTimer(&timer_step2_one);

    /*
    DPCT1049:55: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:79: 'MAX_THREADS_BLOCK' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<float, 1> s_left_scratch_acc_ct1(
          sycl::range<1>(256 /*MAX_THREADS_BLOCK*/), cgh);
      /*
      DPCT1101:80: 'MAX_THREADS_BLOCK' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<float, 1> s_right_scratch_acc_ct1(
          sycl::range<1>(256 /*MAX_THREADS_BLOCK*/), cgh);
      sycl::local_accessor<unsigned int, 0> converged_all_threads_acc_ct1(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(grid_onei * threads_onei, threads_onei),
          [=](sycl::nd_item<3> item_ct1) {
            bisectKernelLarge_OneIntervals(
                input.g_a, input.g_b, mat_size, num_one_intervals,
                result.g_left_one, result.g_right_one, result.g_pos_one,
                precision, item_ct1,
                s_left_scratch_acc_ct1
                    .get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                s_right_scratch_acc_ct1
                    .get_multi_ptr<sycl::access::decorated::no>()
                    .get(),
                converged_all_threads_acc_ct1);
          });
    });

    getLastCudaError("bisectKernelLarge_OneIntervals() FAILED.");
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    sdkStopTimer(&timer_step2_one);

    // process intervals that contained more than one eigenvalue after
    // the first processing step

    // get the number of blocks of intervals that contain, in total when
    // each interval contains only one eigenvalue, not more than
    // MAX_THREADS_BLOCK threads
    unsigned int num_blocks_mult = 0;
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(&num_blocks_mult, result.g_num_blocks_mult,
                                     sizeof(unsigned int))
                             .wait()));

    // setup the execution environment
    sycl::range<3> grid_mult(1, 1, num_blocks_mult);
    sycl::range<3> threads_mult(1, 1, MAX_THREADS_BLOCK);

    sdkStartTimer(&timer_step2_mult);

    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:81: '2 * MAX_THREADS_BLOCK' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<float, 1> s_left_acc_ct1(
          sycl::range<1>(512 /*2 * MAX_THREADS_BLOCK*/), cgh);
      /*
      DPCT1101:82: '2 * MAX_THREADS_BLOCK' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<float, 1> s_right_acc_ct1(
          sycl::range<1>(512 /*2 * MAX_THREADS_BLOCK*/), cgh);
      /*
      DPCT1101:83: '2 * MAX_THREADS_BLOCK' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<unsigned int, 1> s_left_count_acc_ct1(
          sycl::range<1>(512 /*2 * MAX_THREADS_BLOCK*/), cgh);
      /*
      DPCT1101:84: '2 * MAX_THREADS_BLOCK' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<unsigned int, 1> s_right_count_acc_ct1(
          sycl::range<1>(512 /*2 * MAX_THREADS_BLOCK*/), cgh);
      /*
      DPCT1101:85: '2 * MAX_THREADS_BLOCK + 1' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<unsigned int, 1> s_compaction_list_acc_ct1(
          sycl::range<1>(513 /*2 * MAX_THREADS_BLOCK + 1*/), cgh);
      sycl::local_accessor<unsigned int, 0> all_threads_converged_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> num_threads_active_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> num_threads_compaction_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> compact_second_chunk_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> c_block_start_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> c_block_end_acc_ct1(cgh);
      sycl::local_accessor<unsigned int, 0> c_block_offset_output_acc_ct1(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(grid_mult * threads_mult, threads_mult),
          [=](sycl::nd_item<3> item_ct1) {
            bisectKernelLarge_MultIntervals(
                input.g_a, input.g_b, mat_size, result.g_blocks_mult,
                result.g_blocks_mult_sum, result.g_left_mult,
                result.g_right_mult, result.g_left_count_mult,
                result.g_right_count_mult, result.g_lambda_mult,
                result.g_pos_mult, precision, item_ct1,
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
                all_threads_converged_acc_ct1, num_threads_active_acc_ct1,
                num_threads_compaction_acc_ct1, compact_second_chunk_acc_ct1,
                c_block_start_acc_ct1, c_block_end_acc_ct1,
                c_block_offset_output_acc_ct1);
          });
    });

    getLastCudaError("bisectKernelLarge_MultIntervals() FAILED.");
    checkCudaErrors(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
    sdkStopTimer(&timer_step2_mult);
  }

  sdkStopTimer(&timer_total);

  printf("Average time step 1: %f ms\n",
         sdkGetTimerValue(&timer_step1) / (float)iterations);
  printf("Average time step 2, one intervals: %f ms\n",
         sdkGetTimerValue(&timer_step2_one) / (float)iterations);
  printf("Average time step 2, mult intervals: %f ms\n",
         sdkGetTimerValue(&timer_step2_mult) / (float)iterations);

  printf("Average time TOTAL: %f ms\n",
         sdkGetTimerValue(&timer_total) / (float)iterations);

  sdkDeleteTimer(&timer_step1);
  sdkDeleteTimer(&timer_step2_one);
  sdkDeleteTimer(&timer_step2_mult);
  sdkDeleteTimer(&timer_total);
}

////////////////////////////////////////////////////////////////////////////////
//! Process the result, that is obtain result from device and do simple sanity
//! checking
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  filename  output filename
////////////////////////////////////////////////////////////////////////////////
bool processResultDataLargeMatrix(const InputData &input,
                                  const ResultDataLarge &result,
                                  const unsigned int mat_size,
                                  const char *filename,
                                  const unsigned int user_defined,
                                  char *exec_path) {
  bool bCompareResult = false;
  const unsigned int mat_size_ui = sizeof(unsigned int) * mat_size;
  const unsigned int mat_size_f = sizeof(float) * mat_size;

  // copy data from intervals that contained more than one eigenvalue after
  // the first processing step
  float *lambda_mult = (float *)malloc(sizeof(float) * mat_size);
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(lambda_mult, result.g_lambda_mult, sizeof(float) * mat_size)
          .wait()));
  unsigned int *pos_mult =
      (unsigned int *)malloc(sizeof(unsigned int) * mat_size);
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(pos_mult, result.g_pos_mult, sizeof(unsigned int) * mat_size)
          .wait()));

  unsigned int *blocks_mult_sum =
      (unsigned int *)malloc(sizeof(unsigned int) * mat_size);
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(blocks_mult_sum, result.g_blocks_mult_sum,
                                   sizeof(unsigned int) * mat_size)
                           .wait()));

  unsigned int num_one_intervals;
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(&num_one_intervals, result.g_num_one, sizeof(unsigned int))
          .wait()));

  unsigned int sum_blocks_mult = mat_size - num_one_intervals;

  // copy data for intervals that contained one eigenvalue after the first
  // processing step
  float *left_one = (float *)malloc(mat_size_f);
  float *right_one = (float *)malloc(mat_size_f);
  unsigned int *pos_one = (unsigned int *)malloc(mat_size_ui);
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(left_one, result.g_left_one, mat_size_f)
                           .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(right_one, result.g_right_one, mat_size_f)
                           .wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(pos_one, result.g_pos_one, mat_size_ui)
                           .wait()));

  // extract eigenvalues
  float *eigenvals = (float *)malloc(mat_size_f);

  // singleton intervals generated in the second step
  for (unsigned int i = 0; i < sum_blocks_mult; ++i) {
    eigenvals[pos_mult[i] - 1] = lambda_mult[i];
  }

  // singleton intervals generated in the first step
  unsigned int index = 0;

  for (unsigned int i = 0; i < num_one_intervals; ++i, ++index) {
    eigenvals[pos_one[i] - 1] = left_one[i];
  }

  if (1 == user_defined) {
    // store result
    writeTridiagSymMatlab(filename, input.a, input.b + 1, eigenvals, mat_size);
    // getLastCudaError( sdkWriteFilef( filename, eigenvals, mat_size, 0.0f));

    printf("User requests non-default argument(s), skipping self-check!\n");
    bCompareResult = true;
  } else {
    // compare with reference solution

    float *reference = NULL;
    unsigned int input_data_size = 0;

    char *ref_path = sdkFindFilePath("reference.dat", exec_path);
    assert(NULL != ref_path);
    sdkReadFile(ref_path, &reference, &input_data_size, false);
    assert(input_data_size == mat_size);

    // there's an imprecision of Sturm count computation which makes an
    // additional offset necessary
    float tolerance = 1.0e-5f + 5.0e-6f;

    if (sdkCompareL2fe(reference, eigenvals, mat_size, tolerance) == true) {
      bCompareResult = true;
    } else {
      bCompareResult = false;
    }

    free(ref_path);
    free(reference);
  }

  freePtr(eigenvals);
  freePtr(lambda_mult);
  freePtr(pos_mult);
  freePtr(blocks_mult_sum);
  freePtr(left_one);
  freePtr(right_one);
  freePtr(pos_one);

  return bCompareResult;
}
