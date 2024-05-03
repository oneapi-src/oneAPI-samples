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

/* Determine eigenvalues for large matrices for intervals that contained after
 * the first step one eigenvalue
 */

#ifndef _BISECT_KERNEL_LARGE_ONEI_H_
#define _BISECT_KERNEL_LARGE_ONEI_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// includes, project
#include "config.h"
#include "util.h"

// additional kernel
#include "bisect_util.dp.cpp"

////////////////////////////////////////////////////////////////////////////////
//! Determine eigenvalues for large matrices for intervals that after
//! the first step contained one eigenvalue
//! @param  g_d  diagonal elements of symmetric, tridiagonal matrix
//! @param  g_s  superdiagonal elements of symmetric, tridiagonal matrix
//! @param  n    matrix size
//! @param  num_intervals  total number of intervals containing one eigenvalue
//!                         after the first step
//! @param g_left  left interval limits
//! @param g_right  right interval limits
//! @param g_pos  index of interval / number of intervals that are smaller than
//!               right interval limit
//! @param  precision  desired precision of eigenvalues
////////////////////////////////////////////////////////////////////////////////
void bisectKernelLarge_OneIntervals(
    float *g_d, float *g_s, const unsigned int n, unsigned int num_intervals,
    float *g_left, float *g_right, unsigned int *g_pos, float precision,
    const sycl::nd_item<3> &item_ct1, float *s_left_scratch,
    float *s_right_scratch, unsigned int &converged_all_threads) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
  const unsigned int gtid =
      (item_ct1.get_local_range(2) * item_ct1.get_group(2)) +
      item_ct1.get_local_id(2);

  // active interval of thread
  // left and right limit of current interval
  float left, right;
  // number of threads smaller than the right limit (also corresponds to the
  // global index of the eigenvalues contained in the active interval)
  unsigned int right_count;
  // flag if current thread converged
  unsigned int converged = 0;
  // midpoint when current interval is subdivided
  float mid = 0.0f;
  // number of eigenvalues less than mid
  unsigned int mid_count = 0;

  // read data from global memory
  if (gtid < num_intervals) {
    left = g_left[gtid];
    right = g_right[gtid];
    right_count = g_pos[gtid];
  }

  // flag to determine if all threads converged to eigenvalue

  // initialized shared flag
  if (0 == item_ct1.get_local_id(2)) {
    converged_all_threads = 0;
  }

  /*
  DPCT1065:42: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // process until all threads converged to an eigenvalue
  // while( 0 == converged_all_threads) {
  while (true) {
    dpct::atomic_exchange<sycl::access::address_space::generic_space>(
        &converged_all_threads, 1);

    // update midpoint for all active threads
    if ((gtid < num_intervals) && (0 == converged)) {
      mid = computeMidpoint(left, right);
    }

    // find number of eigenvalues that are smaller than midpoint
    mid_count = computeNumSmallerEigenvalsLarge(
        g_d, g_s, n, mid, gtid, num_intervals, s_left_scratch, s_right_scratch,
        converged, cta, item_ct1);

    /*
    DPCT1065:44: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // for all active threads
    if ((gtid < num_intervals) && (0 == converged)) {
      // udpate intervals -- always one child interval survives
      if (right_count == mid_count) {
        right = mid;
      } else {
        left = mid;
      }

      // check for convergence
      float t0 = right - left;
      float t1 = sycl::max(sycl::fabs(right), sycl::fabs(left)) * precision;

      if (t0 < sycl::min(precision, t1)) {
        float lambda = computeMidpoint(left, right);
        left = lambda;
        right = lambda;

        converged = 1;
      } else {
        dpct::atomic_exchange<sycl::access::address_space::generic_space>(
            &converged_all_threads, 0);
      }
    }

    /*
    DPCT1065:45: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (1 == converged_all_threads) {
      break;
    }

    /*
    DPCT1065:46: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // write data back to global memory
  /*
  DPCT1065:43: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (gtid < num_intervals) {
    // intervals converged so left and right interval limit are both identical
    // and identical to the eigenvalue
    g_left[gtid] = left;
  }
}

#endif  // #ifndef _BISECT_KERNEL_LARGE_ONEI_H_
