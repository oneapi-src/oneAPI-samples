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

#ifndef MONTECARLO_REDUCTION_CUH
#define MONTECARLO_REDUCTION_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

////////////////////////////////////////////////////////////////////////////////
// This function calculates total sum for each of the two input arrays.
// SUM_N must be power of two
// Unrolling provides a bit of a performance improvement for small
// to medium path counts.
////////////////////////////////////////////////////////////////////////////////

template <class T, int SUM_N, int blockSize>
void sumReduce(T *sum, T *sum2, sycl::group<3> &cta, sycl::sub_group &tile32,
               __TOptionValue *d_CallValue, const sycl::nd_item<3> &item_ct1) {
  const int VEC = 32;
  const int tid = item_ct1.get_local_linear_id();

  T beta = sum[tid];
  T beta2 = sum2[tid];
  T temp, temp2;

  for (int i = VEC / 2; i > 0; i >>= 1) {
    if (item_ct1.get_sub_group().get_local_linear_id() < i) {
      temp = sum[tid + i];
      temp2 = sum2[tid + i];
      beta += temp;
      beta2 += temp2;
      sum[tid] = beta;
      sum2[tid] = beta2;
    }
    /*
    DPCT1065:2: Consider replacing sycl::sub_group::barrier() with
    sycl::sub_group::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.get_sub_group().barrier();
  }
  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (tid == 0) {
    beta = 0;
    beta2 = 0;
    for (int i = 0; i < item_ct1.get_local_range(2); i += VEC) {
      beta += sum[i];
      beta2 += sum2[i];
    }
    __TOptionValue t = {beta, beta2};
    *d_CallValue = t;
  }
  /*
  DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
}

#endif
