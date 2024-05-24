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
#include <taskflow/sycl/syclflow.hpp>

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

  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  for (int k = 0, i = item_ct1.get_group(2) * ROWS_PER_CTA;
       (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
    double rowThreadSum = 0.0;
    for (int j = item_ct1.get_local_id(2); j < N_ROWS;
         j += item_ct1.get_local_range(2)) {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    for (int offset = tile32.get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      rowThreadSum += sycl::shift_group_left(tile32,
                                             rowThreadSum, offset);
    }
    
    if (tile32.get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
  }

  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < ROWS_PER_CTA) {
    dpct::experimental::logical_group tile8 = dpct::experimental::logical_group(
        item_ct1, cta, ROWS_PER_CTA);
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

  for (int offset = tile32.get_local_linear_range() / 2;
       offset > 0; offset /= 2) {
    sum += sycl::shift_group_left(tile32, sum, offset);
  }

  if (tile32.get_local_linear_id() == 0) {
    warpSum[item_ct1.get_local_id(2) /
            tile32.get_local_range().get(0)] = sum;
  }

  item_ct1.barrier();

  double blockSum = 0.0;
  if (item_ct1.get_local_id(2) <
      (item_ct1.get_local_range(2) /
       tile32.get_local_range().get(0))) {
    blockSum = warpSum[item_ct1.get_local_id(2)];
  }

  if (item_ct1.get_local_id(2) < 32) {
    for (int offset = tile32.get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      blockSum +=
          sycl::shift_group_left(tile32, blockSum, offset);
    }
    
    if (tile32.get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          g_sum, blockSum);
    }
  }
}

double JacobiMethodGpuCudaGraphExecKernelSetParams(
    const float *A, const double *b, const float conv_threshold,
    const int max_iter, double *x, double *x_new, sycl::queue q) {
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);

  tf::Taskflow tflow;
  tf::Executor exe;

  double sum = 0.0;
  double *d_sum = NULL;
  double *params[] = {x, x_new};
  DPCT_CHECK_ERROR(
      d_sum = sycl::malloc_device<double>(1, q));
  int k = 0;

  tf::Task syclDeviceTasks =
      tflow
          .emplace_on(
              [&](tf::syclFlow &sf) {
                tf::syclTask dsum_memset =
                    sf.memset(d_sum, 0, sizeof(double)).name("dsum_memset");

                tf::syclTask jM_kernel =
                    sf.on([=](sycl::handler &cgh) {
                        sycl::local_accessor<double, 1> x_shared_acc_ct1(
                            sycl::range<1>(N_ROWS), cgh);

                        sycl::local_accessor<double, 1> b_shared_acc_ct1(
                            sycl::range<1>(ROWS_PER_CTA + 1), cgh);
                        cgh.parallel_for(
                            sycl::nd_range<3>(nblocks * nthreads, nthreads),
                            [=](sycl::nd_item<3> item_ct1) [
                                [intel::reqd_sub_group_size(32)]] {
                              JacobiMethod(A, b, conv_threshold, params[k % 2],
                                           params[(k + 1) % 2], d_sum, item_ct1,
                                           x_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  b_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
                            });
                      }).name("jacobi_kernel");

                tf::syclTask sum_d2h =
                    sf.memcpy(&sum, d_sum, sizeof(double)).name("sum_d2h");
                q.wait();

                jM_kernel.succeed(dsum_memset).precede(sum_d2h);
              },
              q)
          .name("syclTasks");

  for (k = 0; k < max_iter; k++) {
    exe.run(tflow).wait();

    if (sum <= conv_threshold) {
      q.memset(d_sum, 0, sizeof(double));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;

      size_t sharedMemSize =
          ((nthreads[2] / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads), [=
          ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                finalError(x_new, d_sum, item_ct1,
                           dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
              });
        });
      } else {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads), [=
          ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                finalError(x, d_sum, item_ct1,
                           dpct_local_acc_ct1
                                   .get_multi_ptr<sycl::access::decorated::no>()
                                   .get());
              });
        });
      }
      q.memcpy(&sum, d_sum, sizeof(double)).wait();
      printf("Device iterations : %d\n", k + 1);
      printf("Device error : %.3e\n", sum);
      break;
    }
  }
  sycl::free(d_sum, q);
  return sum;
}


double JacobiMethodGpu(const float *A, const double *b,
                       const float conv_threshold, const int max_iter,
                       double *x, double *x_new, sycl::queue q) {
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);

  double sum = 0.0;
  double *d_sum;
  DPCT_CHECK_ERROR(
      d_sum = sycl::malloc_device<double>(1, q));
  int k = 0;

  for (k = 0; k < max_iter; k++) {
    DPCT_CHECK_ERROR(q.memset(d_sum, 0, sizeof(double)));
    if ((k & 1) == 0) {
     q.submit([&](sycl::handler &cgh) {
       
        sycl::local_accessor<double, 1> x_shared_acc_ct1(
            sycl::range<1>(N_ROWS), cgh);
        
        sycl::local_accessor<double, 1> b_shared_acc_ct1(
            sycl::range<1>(ROWS_PER_CTA + 1), cgh);

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
     
      q.submit([&](sycl::handler &cgh) {
        
        sycl::local_accessor<double, 1> x_shared_acc_ct1(
            sycl::range<1>(N_ROWS), cgh);
        
        sycl::local_accessor<double, 1> b_shared_acc_ct1(
            sycl::range<1>(ROWS_PER_CTA + 1), cgh);

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
    DPCT_CHECK_ERROR(q.memcpy(&sum, d_sum, sizeof(double)));
    DPCT_CHECK_ERROR(q.wait());

    if (sum <= conv_threshold) {
      DPCT_CHECK_ERROR(q.memset(d_sum, 0, sizeof(double)));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;
     
      size_t sharedMemSize = ((nthreads[2] / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        
        q.submit([&](sycl::handler &cgh) {
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
        q.submit([&](sycl::handler &cgh) {
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

      DPCT_CHECK_ERROR(q.memcpy(&sum, d_sum, sizeof(double)));
      DPCT_CHECK_ERROR(q.wait());
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  DPCT_CHECK_ERROR(sycl::free(d_sum, q));
  return sum;
}
