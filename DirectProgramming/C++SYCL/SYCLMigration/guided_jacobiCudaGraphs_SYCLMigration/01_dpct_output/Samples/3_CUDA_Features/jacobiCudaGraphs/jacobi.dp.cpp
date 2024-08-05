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
#include "jacobi.h"

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8

#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP >= 600
#else
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

static void JacobiMethod(const float *A, const double *b,
                                    const float conv_threshold, double *x,
                                    double *x_new, double *sum,
                                    const sycl::nd_item<3> &item_ct1,
                                    double *x_shared, double *b_shared) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
    // N_ROWS == n

  for (int i = item_ct1.get_local_id(2); i < N_ROWS;
       i += item_ct1.get_local_range(2)) {
    x_shared[i] = x[i];
  }

  if (item_ct1.get_local_id(2) < ROWS_PER_CTA) {
    int k = item_ct1.get_local_id(2);
    for (int i = k + (item_ct1.get_group(2) * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      b_shared[i % (ROWS_PER_CTA + 1)] = b[i];
    }
  }

  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  for (int k = 0, i = item_ct1.get_group(2) * ROWS_PER_CTA;
       (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
    double rowThreadSum = 0.0;
    for (int j = item_ct1.get_local_id(2); j < N_ROWS;
         j += item_ct1.get_local_range(2)) {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      rowThreadSum += sycl::shift_group_left(item_ct1.get_sub_group(),
                                             rowThreadSum, offset);
    }

    if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
  }

  /*
  DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < ROWS_PER_CTA) {
    dpct::experimental::logical_group tile8 = dpct::experimental::logical_group(
        item_ct1, item_ct1.get_group(), ROWS_PER_CTA);
    double temp_sum = 0.0;

    int k = item_ct1.get_local_id(2);

    for (int i = k + (item_ct1.get_group(2) * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      double dx = b_shared[i % (ROWS_PER_CTA + 1)];
      dx /= A[i * N_ROWS + i];

      x_new[i] = (x_shared[i] + dx);
      temp_sum += sycl::fabs(dx);
    }

    for (int offset = tile8.get_local_linear_range() / 2; offset > 0;
         offset /= 2) {
      temp_sum += dpct::shift_sub_group_left(item_ct1.get_sub_group(), temp_sum,
                                             offset, 8);
    }

    if (tile8.get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          sum, temp_sum);
    }
  }
}

// Thread block size for finalError kernel should be multiple of 32
static void finalError(double *x, double *g_sum,
                       const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
  auto warpSum = (double *)dpct_local;
  double sum = 0.0;

  int globalThreadId = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

  for (int i = globalThreadId; i < N_ROWS;
       i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
    double d = x[i] - 1.0;
    sum += sycl::fabs(d);
  }

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
       offset > 0; offset /= 2) {
    sum += sycl::shift_group_left(item_ct1.get_sub_group(), sum, offset);
  }

  if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
    warpSum[item_ct1.get_local_id(2) /
            item_ct1.get_sub_group().get_local_range().get(0)] = sum;
  }

  /*
  DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  double blockSum = 0.0;
  if (item_ct1.get_local_id(2) <
      (item_ct1.get_local_range(2) /
       item_ct1.get_sub_group().get_local_range().get(0))) {
    blockSum = warpSum[item_ct1.get_local_id(2)];
  }

  if (item_ct1.get_local_id(2) < 32) {
    for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      blockSum +=
          sycl::shift_group_left(item_ct1.get_sub_group(), blockSum, offset);
    }
    if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          g_sum, blockSum);
    }
  }
}

double JacobiMethodGpuCudaGraphExecKernelSetParams(
    const float *A, const double *b, const float conv_threshold,
    const int max_iter, double *x, double *x_new, dpct::queue_ptr stream) {
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec = NULL;

  double sum = 0.0;
  double *d_sum = NULL;
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_sum = sycl::malloc_device<double>(1, dpct::get_in_order_queue())));

  std::vector<cudaGraphNode_t> nodeDependencies;
  cudaGraphNode_t memcpyNode, jacobiKernelNode, memsetNode;
  dpct::pitched_data memcpyParams_from_data_ct1, memcpyParams_to_data_ct1;
  sycl::id<3> memcpyParams_from_pos_ct1(0, 0, 0),
      memcpyParams_to_pos_ct1(0, 0, 0);
  sycl::range<3> memcpyParams_size_ct1(1, 1, 1);
  dpct::memcpy_direction memcpyParams_direction_ct1;
  cudaMemsetParams memsetParams = {0};

  memsetParams.dst = (void *)d_sum;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  // elementSize can be max 4 bytes, so we take sizeof(float) and width=2
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = 2;
  memsetParams.height = 1;

  /*
  DPCT1007:30: Migration of cudaGraphCreate is not supported.
  */
  checkCudaErrors(cudaGraphCreate(&graph, 0));
  /*
  DPCT1007:31: Migration of cudaGraphAddMemsetNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
  nodeDependencies.push_back(memsetNode);

  /*
  DPCT1082:32: Migration of cudaKernelNodeParams type is not supported.
  */
  cudaKernelNodeParams NodeParams0, NodeParams1;
  NodeParams0.func = (void *)JacobiMethod;
  NodeParams0.gridDim = nblocks;
  NodeParams0.blockDim = nthreads;
  NodeParams0.sharedMemBytes = 0;
  void *kernelArgs0[6] = {(void *)&A, (void *)&b,     (void *)&conv_threshold,
                          (void *)&x, (void *)&x_new, (void *)&d_sum};
  NodeParams0.kernelParams = kernelArgs0;
  NodeParams0.extra = NULL;

  /*
  DPCT1007:33: Migration of cudaGraphAddKernelNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddKernelNode(&jacobiKernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &NodeParams0));

  nodeDependencies.clear();
  nodeDependencies.push_back(jacobiKernelNode);

  memcpyParams_from_data_ct1 = (NULL)->to_pitched_data();
  memcpyParams_from_pos_ct1 = sycl::id<3>(0, 0, 0);
  memcpyParams_from_data_ct1 = dpct::pitched_data(d_sum, sizeof(double), 1, 1);
  memcpyParams_to_data_ct1 = (NULL)->to_pitched_data();
  memcpyParams_to_pos_ct1 = sycl::id<3>(0, 0, 0);
  memcpyParams_to_data_ct1 = dpct::pitched_data(&sum, sizeof(double), 1, 1);
  memcpyParams_size_ct1 = sycl::range<3>(sizeof(double), 1, 1);
  memcpyParams_direction_ct1 = dpct::device_to_host;

  /*
  DPCT1007:34: Migration of cudaGraphAddMemcpyNode is not supported.
  */
  checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &memcpyParams));

  /*
  DPCT1007:35: Migration of cudaGraphInstantiate is not supported.
  */
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  NodeParams1.func = (void *)JacobiMethod;
  NodeParams1.gridDim = nblocks;
  NodeParams1.blockDim = nthreads;
  NodeParams1.sharedMemBytes = 0;
  void *kernelArgs1[6] = {(void *)&A,     (void *)&b, (void *)&conv_threshold,
                          (void *)&x_new, (void *)&x, (void *)&d_sum};
  NodeParams1.kernelParams = kernelArgs1;
  NodeParams1.extra = NULL;

  int k = 0;
  for (k = 0; k < max_iter; k++) {
    /*
    DPCT1007:36: Migration of cudaGraphExecKernelNodeSetParams is not supported.
    */
    checkCudaErrors(cudaGraphExecKernelNodeSetParams(
        graphExec, jacobiKernelNode,
        ((k & 1) == 0) ? &NodeParams0 : &NodeParams1));
    /*
    DPCT1007:37: Migration of cudaGraphLaunch is not supported.
    */
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));

    if (sum <= conv_threshold) {
      checkCudaErrors(
          DPCT_CHECK_ERROR(stream->memset(d_sum, 0, sizeof(double))));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;
      /*
      DPCT1083:4: The size of local memory in the migrated code may be different
      from the original code. Check that the allocated memory size in the
      migrated code is correct.
      */
      size_t sharedMemSize = ((nthreads[2] / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads),
              [=](sycl::nd_item<3> item_ct1)
                  [[intel::reqd_sub_group_size(32)]] {
                    finalError(x_new, d_sum, item_ct1,
                               dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                  });
        });
      } else {
        /*
        DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads),
              [=](sycl::nd_item<3> item_ct1)
                  [[intel::reqd_sub_group_size(32)]] {
                    finalError(x, d_sum, item_ct1,
                               dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                  });
        });
      }

      checkCudaErrors(
          DPCT_CHECK_ERROR(stream->memcpy(&sum, d_sum, sizeof(double))));
      checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_sum, dpct::get_in_order_queue())));
  return sum;
}

double JacobiMethodGpuCudaGraphExecUpdate(const float *A, const double *b,
                                          const float conv_threshold,
                                          const int max_iter, double *x,
                                          double *x_new,
                                          dpct::queue_ptr stream) {
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);
  cudaGraph_t graph;
  cudaGraphExec_t graphExec = NULL;

  double sum = 0.0;
  double *d_sum;
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_sum = sycl::malloc_device<double>(1, dpct::get_in_order_queue())));

  int k = 0;
  for (k = 0; k < max_iter; k++) {
    /*
    DPCT1027:38: The call to cudaStreamBeginCapture was replaced with 0 because
    SYCL currently does not support capture operations on queues.
    */
    checkCudaErrors(0);
    checkCudaErrors(DPCT_CHECK_ERROR(stream->memset(d_sum, 0, sizeof(double))));
    if ((k & 1) == 0) {
      /*
      DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:46: 'N_ROWS' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it
        is correct.
        */
        sycl::local_accessor<double, 1> x_shared_acc_ct1(
            sycl::range<1>(512 /*N_ROWS*/), cgh);
        /*
        DPCT1101:47: 'ROWS_PER_CTA + 1' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.
        */
        sycl::local_accessor<double, 1> b_shared_acc_ct1(
            sycl::range<1>(9 /*ROWS_PER_CTA + 1*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(nblocks * nthreads, nthreads),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              JacobiMethod(
                  A, b, conv_threshold, x, x_new, d_sum, item_ct1,
                  x_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  b_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    } else {
      /*
      DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:48: 'N_ROWS' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it
        is correct.
        */
        sycl::local_accessor<double, 1> x_shared_acc_ct1(
            sycl::range<1>(512 /*N_ROWS*/), cgh);
        /*
        DPCT1101:49: 'ROWS_PER_CTA + 1' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.
        */
        sycl::local_accessor<double, 1> b_shared_acc_ct1(
            sycl::range<1>(9 /*ROWS_PER_CTA + 1*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(nblocks * nthreads, nthreads),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              JacobiMethod(
                  A, b, conv_threshold, x_new, x, d_sum, item_ct1,
                  x_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  b_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
    checkCudaErrors(
        DPCT_CHECK_ERROR(stream->memcpy(&sum, d_sum, sizeof(double))));
    /*
    DPCT1027:39: The call to cudaStreamEndCapture was replaced with 0 because
    SYCL currently does not support capture operations on queues.
    */
    checkCudaErrors(0);

    if (graphExec == NULL) {
      /*
      DPCT1007:40: Migration of cudaGraphInstantiate is not supported.
      */
      checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    } else {
      cudaGraphExecUpdateResult updateResult_out;
      /*
      DPCT1007:41: Migration of cudaGraphExecUpdate is not supported.
      */
      checkCudaErrors(
          cudaGraphExecUpdate(graphExec, graph, NULL, &updateResult_out));
      if (updateResult_out != cudaGraphExecUpdateSuccess) {
        if (graphExec != NULL) {
          /*
          DPCT1007:42: Migration of cudaGraphExecDestroy is not supported.
          */
          checkCudaErrors(cudaGraphExecDestroy(graphExec));
        }
        printf("k = %d graph update failed with error - %d\n", k,
               updateResult_out);
        /*
        DPCT1007:43: Migration of cudaGraphInstantiate is not supported.
        */
        checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
      }
    }
    /*
    DPCT1007:44: Migration of cudaGraphLaunch is not supported.
    */
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));

    if (sum <= conv_threshold) {
      checkCudaErrors(
          DPCT_CHECK_ERROR(stream->memset(d_sum, 0, sizeof(double))));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;
      /*
      DPCT1083:9: The size of local memory in the migrated code may be different
      from the original code. Check that the allocated memory size in the
      migrated code is correct.
      */
      size_t sharedMemSize = ((nthreads[2] / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        /*
        DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads),
              [=](sycl::nd_item<3> item_ct1)
                  [[intel::reqd_sub_group_size(32)]] {
                    finalError(x_new, d_sum, item_ct1,
                               dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                  });
        });
      } else {
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads),
              [=](sycl::nd_item<3> item_ct1)
                  [[intel::reqd_sub_group_size(32)]] {
                    finalError(x, d_sum, item_ct1,
                               dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                  });
        });
      }

      checkCudaErrors(
          DPCT_CHECK_ERROR(stream->memcpy(&sum, d_sum, sizeof(double))));
      checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_sum, dpct::get_in_order_queue())));
  return sum;
}

double JacobiMethodGpu(const float *A, const double *b,
                       const float conv_threshold, const int max_iter,
                       double *x, double *x_new, dpct::queue_ptr stream) {
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);

  double sum = 0.0;
  double *d_sum;
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_sum = sycl::malloc_device<double>(1, dpct::get_in_order_queue())));
  int k = 0;

  for (k = 0; k < max_iter; k++) {
    checkCudaErrors(DPCT_CHECK_ERROR(stream->memset(d_sum, 0, sizeof(double))));
    if ((k & 1) == 0) {
      /*
      DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:50: 'N_ROWS' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it
        is correct.
        */
        sycl::local_accessor<double, 1> x_shared_acc_ct1(
            sycl::range<1>(512 /*N_ROWS*/), cgh);
        /*
        DPCT1101:51: 'ROWS_PER_CTA + 1' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.
        */
        sycl::local_accessor<double, 1> b_shared_acc_ct1(
            sycl::range<1>(9 /*ROWS_PER_CTA + 1*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(nblocks * nthreads, nthreads),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              JacobiMethod(
                  A, b, conv_threshold, x, x_new, d_sum, item_ct1,
                  x_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  b_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    } else {
      /*
      DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:52: 'N_ROWS' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it
        is correct.
        */
        sycl::local_accessor<double, 1> x_shared_acc_ct1(
            sycl::range<1>(512 /*N_ROWS*/), cgh);
        /*
        DPCT1101:53: 'ROWS_PER_CTA + 1' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.
        */
        sycl::local_accessor<double, 1> b_shared_acc_ct1(
            sycl::range<1>(9 /*ROWS_PER_CTA + 1*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(nblocks * nthreads, nthreads),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              JacobiMethod(
                  A, b, conv_threshold, x_new, x, d_sum, item_ct1,
                  x_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  b_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
    checkCudaErrors(
        DPCT_CHECK_ERROR(stream->memcpy(&sum, d_sum, sizeof(double))));
    checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));

    if (sum <= conv_threshold) {
      checkCudaErrors(
          DPCT_CHECK_ERROR(stream->memset(d_sum, 0, sizeof(double))));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;
      /*
      DPCT1083:14: The size of local memory in the migrated code may be
      different from the original code. Check that the allocated memory size in
      the migrated code is correct.
      */
      size_t sharedMemSize = ((nthreads[2] / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        /*
        DPCT1049:13: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads),
              [=](sycl::nd_item<3> item_ct1)
                  [[intel::reqd_sub_group_size(32)]] {
                    finalError(x_new, d_sum, item_ct1,
                               dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                  });
        });
      } else {
        /*
        DPCT1049:15: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads),
              [=](sycl::nd_item<3> item_ct1)
                  [[intel::reqd_sub_group_size(32)]] {
                    finalError(x, d_sum, item_ct1,
                               dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
                  });
        });
      }

      checkCudaErrors(
          DPCT_CHECK_ERROR(stream->memcpy(&sum, d_sum, sizeof(double))));
      checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_sum, dpct::get_in_order_queue())));
  return sum;
}
