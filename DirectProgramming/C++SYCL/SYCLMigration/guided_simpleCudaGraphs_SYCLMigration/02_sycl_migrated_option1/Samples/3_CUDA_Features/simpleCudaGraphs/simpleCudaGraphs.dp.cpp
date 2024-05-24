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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <helper_cuda.h>
#include <vector>
#include <chrono>
#include <chrono>
#include <taskflow/sycl/syclflow.hpp>

using Time = std::chrono::steady_clock;
using ms = std::chrono::milliseconds;
using float_ms = std::chrono::duration<float, ms::period>;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3
#define SUB_GRP_SIZE 32

typedef struct callBackData {
  const char *fn_name;
  double *data;
} callBackData_t;

void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       size_t outputSize, const sycl::nd_item<3> &item_ct1,
                       double *tmp) {

  sycl::group<3> cta = item_ct1.get_group();
  size_t globaltid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    temp_sum += (double)inputVec[i];
  }
  tmp[item_ct1.get_local_linear_id()] = temp_sum;

  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  double beta = temp_sum;
  double temp;

  for (int i = tile32.get_local_linear_range() / 2; i > 0;
       i >>= 1) {
    if (tile32.get_local_linear_id() < i) {
      temp = tmp[item_ct1.get_local_linear_id() + i];
      beta += temp;
      tmp[item_ct1.get_local_linear_id()] = beta;
    }
    tile32.barrier();
  }
  
  item_ct1.barrier();

  if (item_ct1.get_local_linear_id() == 0 &&
      item_ct1.get_group(2) < outputSize) {
    beta = 0.0;
    for (int i = 0; i < item_ct1.get_group().get_local_linear_range();
         i += tile32.get_local_linear_range()) {
      beta += tmp[i];
    }
    outputVec[item_ct1.get_group(2)] = beta;
  }
}

void reduceFinal(double *inputVec, double *result,
                            size_t inputSize, const sycl::nd_item<3> &item_ct1,
                            double *tmp) {

  sycl::group<3> cta = item_ct1.get_group();
  size_t globaltid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    temp_sum += (double)inputVec[i];
  }
  tmp[item_ct1.get_local_linear_id()] = temp_sum;

  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  // do reduction in shared mem
  if ((item_ct1.get_local_range(2) >= 512) &&
      (item_ct1.get_local_linear_id() < 256)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 256];
  }

  item_ct1.barrier();

  if ((item_ct1.get_local_range(2) >= 256) &&
      (item_ct1.get_local_linear_id() < 128)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 128];
  }

  item_ct1.barrier();

  if ((item_ct1.get_local_range(2) >= 128) &&
      (item_ct1.get_local_linear_id() < 64)) {
    tmp[item_ct1.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item_ct1.get_local_linear_id() + 64];
  }

  item_ct1.barrier();

  if (item_ct1.get_local_linear_id() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (item_ct1.get_local_range(2) >= 64) temp_sum +=
        tmp[item_ct1.get_local_linear_id() + 32];
    // Reduce final warp using shuffle
    for (int offset =tile32.get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      temp_sum +=
          sycl::shift_group_left(tile32, temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (item_ct1.get_local_linear_id() == 0) result[0] = temp_sum;
}

void init_input(float *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void myHostNodeCallback(void *data) {
  // Check status of GPU after stream operations are done
  callBackData_t *tmp = (callBackData_t *)(data);

  double *result = (double *)(tmp->data);
  char *function = (char *)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  *result = 0.0;  // reset the result
}

void syclTaskFlowManual(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                      double *result_d, size_t inputSize, size_t numOfBlocks, sycl::queue q_ct1) {
  tf::Taskflow tflow;
  tf::Executor exe;

  double result_h = 0.0;
  size_t sf_Task = 0, tf_Task = 0;

  tf::Task syclKernelTask =
      tflow
          .emplace_on(
              [&](tf::syclFlow &sf) {
                tf::syclTask inputVec_h2d =
                    sf.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize)
                        .name("inputVec_h2d");
                tf::syclTask outputVec_memset =
                    sf.memset(outputVec_d, 0, numOfBlocks * sizeof(double))
                        .name("outputVecd_memset");

                tf::syclTask reduce_kernel =
                    sf.on([=](sycl::handler &cgh) {
                        sycl::local_accessor<double, 1> tmp(
                            sycl::range<1>(THREADS_PER_BLOCK), cgh);
                        cgh.parallel_for(
                            sycl::nd_range<3>{
                                sycl::range<3>(1, 1, numOfBlocks) *
                                    sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                                sycl::range<3>(1, 1, THREADS_PER_BLOCK)},
                            [=](sycl::nd_item<3> item_ct1)
                                [[intel::reqd_sub_group_size(SUB_GRP_SIZE)]] {
                                  reduce(inputVec_d, outputVec_d, inputSize,
                                         numOfBlocks, item_ct1,
                                    
tmp.get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                                });
                      }).name("reduce_kernel");

                tf::syclTask resultd_memset =
                    sf.memset(result_d, 0, sizeof(double))
                        .name("resultd_memset");

                tf::syclTask reduceFinal_kernel =
                    sf.on([=](sycl::handler &cgh) {
                        sycl::local_accessor<double, 1> tmp(
                            sycl::range<1>(THREADS_PER_BLOCK), cgh);
                        cgh.parallel_for(
                            sycl::nd_range<3>{
                                sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                                sycl::range<3>(1, 1, THREADS_PER_BLOCK)},
                            [=](sycl::nd_item<3> item_ct1)
                                [[intel::reqd_sub_group_size(SUB_GRP_SIZE)]] {
                                  reduceFinal(outputVec_d, result_d,
                                              numOfBlocks, item_ct1,
                                              tmp.get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                                });
                      }).name("reduceFinal_kernel");

                tf::syclTask result_d2h =
                    sf.memcpy(&result_h, result_d, sizeof(double))
                        .name("resulth_d2h");

                reduce_kernel.succeed(inputVec_h2d, outputVec_memset)
                    .precede(reduceFinal_kernel);
                reduceFinal_kernel.succeed(resultd_memset).precede(result_d2h);

                sf_Task = sf.num_tasks();
              },
              q_ct1)
          .name("syclKernelTask");

  callBackData_t hostFnData = {0};
  hostFnData.data = &result_h;
  hostFnData.fn_name = "syclTaskFlowManual";

  tf::Task syclHostTask =
      tflow.emplace([&]() { myHostNodeCallback(&hostFnData); })
          .name("syclHostTask");

  syclHostTask.succeed(syclKernelTask);

  tf_Task = tflow.num_tasks() - 1;

  exe.run_n(tflow, GRAPH_LAUNCH_ITERATIONS)
      .wait();  // launch & runs the tflow 3 times

  printf(
      "\nNumber of tasks(nodes) in the syclTaskFlow(graph) created manually = "
      "%zu\n",
      sf_Task + tf_Task);

  printf("Cloned Graph Output.. \n");
  tf::Taskflow tflow_clone(std::move(tflow));
  exe.run_n(tflow_clone, GRAPH_LAUNCH_ITERATIONS).wait();
}

int main(int argc, char **argv) {

  sycl::queue q_ct1{aspect_selector(sycl::aspect::fp64)};
  std::cout << "Device: "
            << q_ct1.get_device().get_info<sycl::info::device::name>() << "\n";

  size_t size = 1 << 24;  // number of elements to reduce
  size_t maxBlocks = 512;

  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;

  DPCT_CHECK_ERROR(
      inputVec_h = sycl::malloc_host<float>(size, q_ct1));
  DPCT_CHECK_ERROR(inputVec_d = sycl::malloc_device<float>(
                                       size, q_ct1));
  DPCT_CHECK_ERROR(outputVec_d = sycl::malloc_device<double>(
                                       maxBlocks, q_ct1));
  DPCT_CHECK_ERROR(
      result_d = sycl::malloc_device<double>(1, q_ct1));

  init_input(inputVec_h, size);

  auto startTimer1 = Time::now();
  syclTaskFlowManual(inputVec_h, inputVec_d, outputVec_d, result_d, size,
                   maxBlocks, q_ct1);
  auto stopTimer1 = Time::now();
  auto Timer_duration1 =
      std::chrono::duration_cast<float_ms>(stopTimer1 - startTimer1).count();
  printf("Elapsed Time of SYCL Taskflow Manual : %f (ms)\n", Timer_duration1);

  DPCT_CHECK_ERROR(sycl::free(inputVec_d, q_ct1));
  DPCT_CHECK_ERROR(sycl::free(outputVec_d, q_ct1));
  DPCT_CHECK_ERROR(sycl::free(result_d, q_ct1));
  DPCT_CHECK_ERROR(sycl::free(inputVec_h, q_ct1));
  return EXIT_SUCCESS;
}
