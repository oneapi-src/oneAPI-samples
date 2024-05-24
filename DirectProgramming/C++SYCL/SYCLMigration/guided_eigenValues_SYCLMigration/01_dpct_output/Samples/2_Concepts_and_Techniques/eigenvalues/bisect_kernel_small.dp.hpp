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

/* Determine eigenvalues for small symmetric, tridiagonal matrix */

#ifndef _BISECT_KERNEL_SMALL_H_
#define _BISECT_KERNEL_SMALL_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// includes, project
#include "config.h"
#include "util.h"

// additional kernel
#include "bisect_util.dp.cpp"

////////////////////////////////////////////////////////////////////////////////
//! Bisection to find eigenvalues of a real, symmetric, and tridiagonal matrix
//! @param  g_d  diagonal elements in global memory
//! @param  g_s  superdiagonal elements in global elements (stored so that the
//!              element *(g_s - 1) can be accessed an equals 0
//! @param  n   size of matrix
//! @param  lg  lower bound of input interval (e.g. Gerschgorin interval)
//! @param  ug  upper bound of input interval (e.g. Gerschgorin interval)
//! @param  lg_eig_count  number of eigenvalues that are smaller than \a lg
//! @param  lu_eig_count  number of eigenvalues that are smaller than \a lu
//! @param  epsilon  desired accuracy of eigenvalues to compute
////////////////////////////////////////////////////////////////////////////////
void bisectKernel(float *g_d, float *g_s, const unsigned int n,
                             float *g_left, float *g_right,
                             unsigned int *g_left_count,
                             unsigned int *g_right_count, const float lg,
                             const float ug, const unsigned int lg_eig_count,
                             const unsigned int ug_eig_count, float epsilon,
                             const sycl::nd_item<3> &item_ct1, float *s_left,
                             float *s_right, unsigned int *s_left_count,
                             unsigned int *s_right_count,
                             unsigned int *s_compaction_list,
                             unsigned int &compact_second_chunk,
                             unsigned int &all_threads_converged,
                             unsigned int &num_threads_active,
                             unsigned int &num_threads_compaction) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
  // intervals (store left and right because the subdivision tree is in general
  // not dense

  // number of eigenvalues that are smaller than s_left / s_right
  // (correspondence is realized via indices)

  // helper for stream compaction

  // state variables for whole block
  // if 0 then compaction of second chunk of child intervals is not necessary
  // (because all intervals had exactly one non-dead child)

  // number of currently active threads

  // number of threads to use for stream compaction

  // helper for exclusive scan
  unsigned int *s_compaction_list_exc = s_compaction_list + 1;

  // variables for currently processed interval
  // left and right limit of active interval
  float left = 0.0f;
  float right = 0.0f;
  unsigned int left_count = 0;
  unsigned int right_count = 0;
  // midpoint of active interval
  float mid = 0.0f;
  // number of eigenvalues smaller then mid
  unsigned int mid_count = 0;
  // affected from compaction
  unsigned int is_active_second = 0;

  s_compaction_list[item_ct1.get_local_id(2)] = 0;
  s_left[item_ct1.get_local_id(2)] = 0;
  s_right[item_ct1.get_local_id(2)] = 0;
  s_left_count[item_ct1.get_local_id(2)] = 0;
  s_right_count[item_ct1.get_local_id(2)] = 0;

  /*
  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // set up initial configuration
  if (0 == item_ct1.get_local_id(2)) {
    s_left[0] = lg;
    s_right[0] = ug;
    s_left_count[0] = lg_eig_count;
    s_right_count[0] = ug_eig_count;

    compact_second_chunk = 0;
    num_threads_active = 1;

    num_threads_compaction = 1;
  }

  // for all active threads read intervals from the last level
  // the number of (worst case) active threads per level l is 2^l
  while (true) {
    all_threads_converged = 1;
    /*
    DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    is_active_second = 0;
    subdivideActiveInterval(item_ct1.get_local_id(2), s_left, s_right,
                            s_left_count, s_right_count, num_threads_active,
                            left, right, left_count, right_count, mid,
                            all_threads_converged);

    /*
    DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // check if done
    if (1 == all_threads_converged) {
      break;
    }

    /*
    DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // compute number of eigenvalues smaller than mid
    // use all threads for reading the necessary matrix data from global
    // memory
    // use s_left and s_right as scratch space for diagonal and
    // superdiagonal of matrix
    mid_count = computeNumSmallerEigenvals(
        g_d, g_s, n, mid, item_ct1.get_local_id(2), num_threads_active, s_left,
        s_right, (left == right), cta, item_ct1);

    /*
    DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // store intervals
    // for all threads store the first child interval in a continuous chunk of
    // memory, and the second child interval -- if it exists -- in a second
    // chunk; it is likely that all threads reach convergence up to
    // \a epsilon at the same level; furthermore, for higher level most / all
    // threads will have only one child, storing the first child compactly will
    // (first) avoid to perform a compaction step on the first chunk, (second)
    // make it for higher levels (when all threads / intervals have
    // exactly one child)  unnecessary to perform a compaction of the second
    // chunk
    if (item_ct1.get_local_id(2) < num_threads_active) {
      if (left != right) {
        // store intervals
        storeNonEmptyIntervals(
            item_ct1.get_local_id(2), num_threads_active, s_left, s_right,
            s_left_count, s_right_count, left, mid, right, left_count,
            mid_count, right_count, epsilon, compact_second_chunk,
            s_compaction_list_exc, is_active_second, item_ct1);
      } else {
        storeIntervalConverged(s_left, s_right, s_left_count, s_right_count,
                               left, mid, right, left_count, mid_count,
                               right_count, s_compaction_list_exc,
                               compact_second_chunk, num_threads_active,
                               is_active_second, item_ct1);
      }
    }

    // necessary so that compact_second_chunk is up-to-date
    /*
    DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // perform compaction of chunk where second children are stored
    // scan of (num_threads_active / 2) elements, thus at most
    // (num_threads_active / 4) threads are needed
    if (compact_second_chunk > 0) {
      createIndicesCompaction(s_compaction_list_exc, num_threads_compaction,
                              cta, item_ct1);

      compactIntervals(s_left, s_right, s_left_count, s_right_count, mid, right,
                       mid_count, right_count, s_compaction_list,
                       num_threads_active, is_active_second, item_ct1);
    }

    /*
    DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (0 == item_ct1.get_local_id(2)) {
      // update number of active threads with result of reduction
      num_threads_active += s_compaction_list[num_threads_active];

      num_threads_compaction = ceilPow2(num_threads_active);

      compact_second_chunk = 0;
    }

    /*
    DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  /*
  DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // write resulting intervals to global mem
  // for all threads write if they have been converged to an eigenvalue to
  // a separate array

  // at most n valid intervals
  if (item_ct1.get_local_id(2) < n) {
    // intervals converged so left and right limit are identical
    g_left[item_ct1.get_local_id(2)] = s_left[item_ct1.get_local_id(2)];
    // left count is sufficient to have global order
    g_left_count[item_ct1.get_local_id(2)] =
        s_left_count[item_ct1.get_local_id(2)];
  }
}

#endif  // #ifndef _BISECT_KERNEL_SMALL_H_
