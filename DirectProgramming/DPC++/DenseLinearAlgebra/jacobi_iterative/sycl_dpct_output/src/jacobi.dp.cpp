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


#include <CL/sycl.hpp>
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
                                    sycl::nd_item<3> item_ct1, double *x_shared,
                                    double *b_shared) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();
    // N_ROWS == n

  for (int i = item_ct1.get_local_id(2); i < N_ROWS;
       i += item_ct1.get_local_range().get(2)) {
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
  DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  /*
  DPCT1007:20: Migration of this CUDA API is not supported by the Intel(R) DPC++
  Compatibility Tool.
  */
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

   // auto subwarp = cooperative_groups::tiled_partition<16>(cooperative_groups::this_thread_block());

  for (int k = 0, i = item_ct1.get_group(2) * ROWS_PER_CTA;
       (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
    double rowThreadSum = 0.0;
    for (int j = item_ct1.get_local_id(2); j < N_ROWS;
         j += item_ct1.get_local_range().get(2)) {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      rowThreadSum += tile32.shfl_down(rowThreadSum, offset);
    }

    if (tile32.thread_rank() == 0) {
      dpct::atomic_fetch_add<double, sycl::access::address_space::local_space>(
          &b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
  }

  /*
  DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < ROWS_PER_CTA) {
    cg::thread_block_tile<ROWS_PER_CTA> tile8 =
        /*
        DPCT1007:21: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        cg::tiled_partition<ROWS_PER_CTA>(cta);
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

    for (int offset = tile8.size() / 2; offset > 0; offset /= 2) {
      temp_sum += tile8.shfl_down(temp_sum, offset);
    }

    if (tile8.thread_rank() == 0) {
      /*
      DPCT1039:22: The generated code assumes that "sum" points to the global
      memory address space. If it points to a local memory address space,
      replace "dpct::atomic_fetch_add" with "dpct::atomic_fetch_add<double,
      sycl::access::address_space::local_space>".
      */
      dpct::atomic_fetch_add(sum, temp_sum);
    }
  }
}

// Thread block size for finalError kernel should be multiple of 32
static void finalError(double *x, double *g_sum, sycl::nd_item<3> item_ct1,
                       uint8_t *dpct_local) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();
  auto warpSum = (double *)dpct_local;
  double sum = 0.0;

  int globalThreadId =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);

  for (int i = globalThreadId; i < N_ROWS;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    double d = x[i] - 1.0;
    sum += sycl::fabs(d);
  }

  /*
  DPCT1007:24: Migration of this CUDA API is not supported by the Intel(R) DPC++
  Compatibility Tool.
  */
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
    sum += tile32.shfl_down(sum, offset);
  }

  if (tile32.thread_rank() == 0) {
    warpSum[item_ct1.get_local_id(2) /
            item_ct1.get_sub_group().get_local_range().get(0)] = sum;
  }

  /*
  DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  double blockSum = 0.0;
  if (item_ct1.get_local_id(2) <
      (item_ct1.get_local_range().get(2) /
       item_ct1.get_sub_group().get_local_range().get(0))) {
    blockSum = warpSum[item_ct1.get_local_id(2)];
  }

  if (item_ct1.get_local_id(2) < 32) {
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      blockSum += tile32.shfl_down(blockSum, offset);
    }
    if (tile32.thread_rank() == 0) {
      /*
      DPCT1039:25: The generated code assumes that "g_sum" points to the global
      memory address space. If it points to a local memory address space,
      replace "dpct::atomic_fetch_add" with "dpct::atomic_fetch_add<double,
      sycl::access::address_space::local_space>".
      */
      dpct::atomic_fetch_add(g_sum, blockSum);
    }
  }
}

double JacobiMethodGpu(const float *A, const double *b,
                       const float conv_threshold, const int max_iter,
                       double *x, double *x_new, sycl::queue *stream) {
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);

  double sum = 0.0;
  double *d_sum;
  d_sum = sycl::malloc_device<double>(1, dpct::get_default_queue());
  int k = 0;

  for (k = 0; k < max_iter; k++) {
    stream->memset(d_sum, 0, sizeof(double));
    if ((k & 1) == 0) {
      /*
      DPCT1049:26: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            x_shared_acc_ct1(sycl::range<1>(512 /*N_ROWS*/), cgh);
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            b_shared_acc_ct1(sycl::range<1>(9 /*ROWS_PER_CTA + 1*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                         [=](sycl::nd_item<3> item_ct1) {
                           JacobiMethod(A, b, conv_threshold, x, x_new, d_sum,
                                        item_ct1,
                                        x_shared_acc_ct1.get_pointer(),
                                        b_shared_acc_ct1.get_pointer());
                         });
      });
    } else {
      /*
      DPCT1049:27: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            x_shared_acc_ct1(sycl::range<1>(512 /*N_ROWS*/), cgh);
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            b_shared_acc_ct1(sycl::range<1>(9 /*ROWS_PER_CTA + 1*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                         [=](sycl::nd_item<3> item_ct1) {
                           JacobiMethod(A, b, conv_threshold, x_new, x, d_sum,
                                        item_ct1,
                                        x_shared_acc_ct1.get_pointer(),
                                        b_shared_acc_ct1.get_pointer());
                         });
      });
    }
    stream->memcpy(&sum, d_sum, sizeof(double));
    stream->wait();

    if (sum <= conv_threshold) {
      stream->memset(d_sum, 0, sizeof(double));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;
      /*
      DPCT1083:29: The size of local memory in the migrated code may be
      different from the original code. Check that the allocated memory size in
      the migrated code is correct.
      */
      size_t sharedMemSize = ((nthreads[2] / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        /*
        DPCT1049:28: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        stream->submit([&](sycl::handler &cgh) {
          sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                         sycl::access::target::local>
              dpct_local_acc_ct1(sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                           [=](sycl::nd_item<3> item_ct1) {
                             finalError(x_new, d_sum, item_ct1,
                                        dpct_local_acc_ct1.get_pointer());
                           });
        });
      } else {
        /*
        DPCT1049:30: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        stream->submit([&](sycl::handler &cgh) {
          sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                         sycl::access::target::local>
              dpct_local_acc_ct1(sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                           [=](sycl::nd_item<3> item_ct1) {
                             finalError(x, d_sum, item_ct1,
                                        dpct_local_acc_ct1.get_pointer());
                           });
        });
      }

      stream->memcpy(&sum, d_sum, sizeof(double));
      stream->wait();
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  sycl::free(d_sum, dpct::get_default_queue());
  return sum;
}
