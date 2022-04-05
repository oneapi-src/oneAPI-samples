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


#include <helper_cuda.h>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "jacobi.h"

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 32
#define SUB_GRP_SIZE 16

static void JacobiMethod(const float *A, const double *b,
                         const float conv_threshold, double *x, double *x_new,
                         double *sum, sycl::nd_item<3> item_ct1,
                         double *x_shared, double *b_shared) {
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

  item_ct1.barrier();

  sycl::sub_group tile_sg = item_ct1.get_sub_group();
  int tile_sg_size = tile_sg.get_local_range().get(0);

  for (int k = 0, i = item_ct1.get_group(2) * ROWS_PER_CTA;
       (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
    double rowThreadSum = 0.0;
    for (int j = item_ct1.get_local_id(2); j < N_ROWS;
         j += item_ct1.get_local_range().get(2)) {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    for (int offset = tile_sg.get_local_range().get(0) / 2; offset > 0;
         offset /= 2) {
      rowThreadSum += tile_sg.shuffle_down(rowThreadSum, offset);
    }

    if (tile_sg.get_local_id()[0] == 0) {
      sycl::ext::oneapi::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::local_space>
          at_h_sum{b_shared[i % (ROWS_PER_CTA + 1)]};
      at_h_sum.fetch_add(-rowThreadSum);
    }
  }

  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < ROWS_PER_CTA) {
    sycl::sub_group tile_sg = item_ct1.get_sub_group();
    int tile_sg_size = tile_sg.get_local_range().get(0);

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

    for (int offset = tile_sg.get_local_range().get(0) / 2; offset > 0;
         offset /= 2) {
      temp_sum += tile_sg.shuffle_down(temp_sum, offset);
    }

    if (tile_sg.get_local_id()[0] == 0) {
      sycl::ext::oneapi::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
          at_sum{*sum};
      at_sum.fetch_add(temp_sum);
    }
  }
}

// Thread block size for finalError kernel should be multiple of 32
static void finalError(double *x, double *g_sum, sycl::nd_item<3> item_ct1,
                       uint8_t *dpct_local) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();
  auto sg_Sum = (double *)dpct_local;
  double sum = 0.0;

  int globalThreadId =
      item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
      item_ct1.get_local_id(2);

  for (int i = globalThreadId; i < N_ROWS;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    double d = x[i] - 1.0;
    sum += sycl::fabs(d);
  }

  sycl::sub_group tile_sg = item_ct1.get_sub_group();
  int tile_sg_size = tile_sg.get_local_range().get(0);

  for (int offset = tile_sg.get_local_range().get(0) / 2; offset > 0;
       offset /= 2) {
    sum += tile_sg.shuffle_down(sum, offset);
  }

  if (tile_sg.get_local_id()[0] == 0) {
    sg_Sum[item_ct1.get_local_id(2) / tile_sg.get_local_range().get(0)] = sum;
  }

  item_ct1.barrier();

  double blockSum = 0.0;
  if (item_ct1.get_local_id(2) <
      (item_ct1.get_local_range().get(2) /
       item_ct1.get_sub_group().get_local_range().get(0))) {
    blockSum = sg_Sum[item_ct1.get_local_id(2)];
  }

  if (item_ct1.get_local_id(2) < SUB_GRP_SIZE) {
    for (int offset = tile_sg.get_local_range().get(0) / 2; offset > 0;
         offset /= 2) {
      blockSum += tile_sg.shuffle_down(blockSum, offset);
    }
    if (tile_sg.get_local_id()[0] == 0) {
      sycl::ext::oneapi::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
          at_d_sum{*g_sum};
      at_d_sum.fetch_add(blockSum);
    }
  }
}

double JacobiMethodGpu(const float *A, const double *b,
                       const float conv_threshold, const int max_iter,
                       double *x, double *x_new, sycl::queue *q) {
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);

  double sum = 0.0;
  double *d_sum;

  d_sum = sycl::malloc_device<double>(1, dpct::get_default_queue());
  int k = 0;

  for (k = 0; k < max_iter; k++) {
    q->memset(d_sum, 0, sizeof(double));
    if ((k & 1) == 0) {
      q->submit([&](sycl::handler &cgh) {
         sycl::accessor<double, 1, sycl::access_mode::read_write,
                        sycl::access::target::local>
             x_shared_acc_ct1(sycl::range<1>(N_ROWS), cgh);
         sycl::accessor<double, 1, sycl::access_mode::read_write,
                        sycl::access::target::local>
             b_shared_acc_ct1(sycl::range<1>(ROWS_PER_CTA + 1), cgh);

         cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                          [=](sycl::nd_item<3> item_ct1)
                              [[intel::reqd_sub_group_size(ROWS_PER_CTA)]] {
                                JacobiMethod(A, b, conv_threshold, x, x_new,
                                             d_sum, item_ct1,
                                             x_shared_acc_ct1.get_pointer(),
                                             b_shared_acc_ct1.get_pointer());
                              });
       }).wait();
    } else {
      q->submit([&](sycl::handler &cgh) {
         sycl::accessor<double, 1, sycl::access_mode::read_write,
                        sycl::access::target::local>
             x_shared_acc_ct1(sycl::range<1>(N_ROWS), cgh);
         sycl::accessor<double, 1, sycl::access_mode::read_write,
                        sycl::access::target::local>
             b_shared_acc_ct1(sycl::range<1>(ROWS_PER_CTA + 1), cgh);

         cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                          [=](sycl::nd_item<3> item_ct1)
                              [[intel::reqd_sub_group_size(ROWS_PER_CTA)]] {
                                JacobiMethod(A, b, conv_threshold, x_new, x,
                                             d_sum, item_ct1,
                                             x_shared_acc_ct1.get_pointer(),
                                             b_shared_acc_ct1.get_pointer());
                              });
       }).wait();
    }

    q->memcpy(&sum, d_sum, sizeof(double)).wait();

    if (sum <= conv_threshold) {
      q->memset(d_sum, 0, sizeof(double));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;
      size_t sharedMemSize =
          ((nthreads[2] / SUB_GRP_SIZE) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        q->submit([&](sycl::handler &cgh) {
           sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                          sycl::access::target::local>
               dpct_local_acc_ct1(sycl::range<1>(sharedMemSize), cgh);

           cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                            [=](sycl::nd_item<3> item_ct1)
                                [[intel::reqd_sub_group_size(SUB_GRP_SIZE)]] {
                                  finalError(x_new, d_sum, item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                });
         }).wait();
      } else {
        q->submit([&](sycl::handler &cgh) {
           sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                          sycl::access::target::local>
               dpct_local_acc_ct1(sycl::range<1>(sharedMemSize), cgh);

           cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                            [=](sycl::nd_item<3> item_ct1)
                                [[intel::reqd_sub_group_size(SUB_GRP_SIZE)]] {
                                  finalError(x, d_sum, item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                });
         }).wait();
      }

      q->memcpy(&sum, d_sum, sizeof(double)).wait();

      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  sycl::free(d_sum, dpct::get_default_queue());
  return sum;
}
