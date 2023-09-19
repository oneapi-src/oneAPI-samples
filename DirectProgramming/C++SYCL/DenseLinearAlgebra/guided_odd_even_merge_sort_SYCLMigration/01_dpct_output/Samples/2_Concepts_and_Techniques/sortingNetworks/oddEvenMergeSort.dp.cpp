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

//Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/oemen.htm

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>

#include <helper_cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.dp.hpp"

////////////////////////////////////////////////////////////////////////////////
// Monolithic Bacther's sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
void oddEvenMergeSortShared(uint *d_DstKey, uint *d_DstVal,
                                       uint *d_SrcKey, uint *d_SrcVal,
                                       uint arrayLength, uint dir,
                                       const sycl::nd_item<3> &item_ct1,
                                       uint *s_key, uint *s_val) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();
  // Shared memory storage for one or more small vectors

  // Offset to the beginning of subbatch and load data
  d_SrcKey +=
      item_ct1.get_group(2) * SHARED_SIZE_LIMIT + item_ct1.get_local_id(2);
  d_SrcVal +=
      item_ct1.get_group(2) * SHARED_SIZE_LIMIT + item_ct1.get_local_id(2);
  d_DstKey +=
      item_ct1.get_group(2) * SHARED_SIZE_LIMIT + item_ct1.get_local_id(2);
  d_DstVal +=
      item_ct1.get_group(2) * SHARED_SIZE_LIMIT + item_ct1.get_local_id(2);
  s_key[item_ct1.get_local_id(2) + 0] = d_SrcKey[0];
  s_val[item_ct1.get_local_id(2) + 0] = d_SrcVal[0];
  s_key[item_ct1.get_local_id(2) + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
  s_val[item_ct1.get_local_id(2) + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

  for (uint size = 2; size <= arrayLength; size <<= 1) {
    uint stride = size / 2;
    uint offset = item_ct1.get_local_id(2) & (stride - 1);

    {
      /*
      DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      uint pos = 2 * item_ct1.get_local_id(2) -
                 (item_ct1.get_local_id(2) & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], dir);
      stride >>= 1;
    }

    for (; stride > 0; stride >>= 1) {
      /*
      DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      uint pos = 2 * item_ct1.get_local_id(2) -
                 (item_ct1.get_local_id(2) & (stride - 1));

      if (offset >= stride)
        Comparator(s_key[pos - stride], s_val[pos - stride], s_key[pos + 0],
                   s_val[pos + 0], dir);
    }
  }

  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  d_DstKey[0] = s_key[item_ct1.get_local_id(2) + 0];
  d_DstVal[0] = s_val[item_ct1.get_local_id(2) + 0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
      s_key[item_ct1.get_local_id(2) + (SHARED_SIZE_LIMIT / 2)];
  d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
      s_val[item_ct1.get_local_id(2) + (SHARED_SIZE_LIMIT / 2)];
}

////////////////////////////////////////////////////////////////////////////////
// Odd-even merge sort iteration kernel
// for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
void oddEvenMergeGlobal(uint *d_DstKey, uint *d_DstVal,
                                   uint *d_SrcKey, uint *d_SrcVal,
                                   uint arrayLength, uint size, uint stride,
                                   uint dir, const sycl::nd_item<3> &item_ct1) {
  uint global_comparatorI =
      item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  // Odd-even merge
  uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

  if (stride < size / 2) {
    uint offset = global_comparatorI & ((size / 2) - 1);

    if (offset >= stride) {
      uint keyA = d_SrcKey[pos - stride];
      uint valA = d_SrcVal[pos - stride];
      uint keyB = d_SrcKey[pos + 0];
      uint valB = d_SrcVal[pos + 0];

      Comparator(keyA, valA, keyB, valB, dir);

      d_DstKey[pos - stride] = keyA;
      d_DstVal[pos - stride] = valA;
      d_DstKey[pos + 0] = keyB;
      d_DstVal[pos + 0] = valB;
    }
  } else {
    uint keyA = d_SrcKey[pos + 0];
    uint valA = d_SrcVal[pos + 0];
    uint keyB = d_SrcKey[pos + stride];
    uint valB = d_SrcVal[pos + stride];

    Comparator(keyA, valA, keyB, valB, dir);

    d_DstKey[pos + 0] = keyA;
    d_DstVal[pos + 0] = valA;
    d_DstKey[pos + stride] = keyB;
    d_DstVal[pos + stride] = valB;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
// Helper function
extern "C" uint factorRadix2(uint *log2L, uint L) {
  if (!L) {
    *log2L = 0;
    return 0;
  } else {
    for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++)
      ;

    return L;
  }
}

extern "C" uint oddEvenMergeSort(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey,
                                 uint *d_SrcVal, uint batchSize,
                                 uint arrayLength, uint dir) {
  // Nothing to sort
  if (arrayLength < 2) return 0;

  // Only power-of-two array lengths are supported by this implementation
  uint log2L;
  uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
  assert(factorizationRemainder == 1);

  dir = (dir != 0);

  uint blockCount = (batchSize * arrayLength) / SHARED_SIZE_LIMIT;
  uint threadCount = SHARED_SIZE_LIMIT / 2;

  if (arrayLength <= SHARED_SIZE_LIMIT) {
    assert(SHARED_SIZE_LIMIT % arrayLength == 0);
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:19: 'SHARED_SIZE_LIMIT' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<uint, 1> s_key_acc_ct1(
          sycl::range<1>(1024 /*SHARED_SIZE_LIMIT*/), cgh);
      /*
      DPCT1101:20: 'SHARED_SIZE_LIMIT' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<uint, 1> s_val_acc_ct1(
          sycl::range<1>(1024 /*SHARED_SIZE_LIMIT*/), cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blockCount) *
                                             sycl::range<3>(1, 1, threadCount),
                                         sycl::range<3>(1, 1, threadCount)),
                       [=](sycl::nd_item<3> item_ct1) {
                         oddEvenMergeSortShared(d_DstKey, d_DstVal, d_SrcKey,
                                                d_SrcVal, arrayLength, dir,
                                                item_ct1,
                                                s_key_acc_ct1.get_pointer(),
                                                s_val_acc_ct1.get_pointer());
                       });
    });
  } else {
    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:21: 'SHARED_SIZE_LIMIT' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<uint, 1> s_key_acc_ct1(
          sycl::range<1>(1024 /*SHARED_SIZE_LIMIT*/), cgh);
      /*
      DPCT1101:22: 'SHARED_SIZE_LIMIT' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments, if
      it is correct.
      */
      sycl::local_accessor<uint, 1> s_val_acc_ct1(
          sycl::range<1>(1024 /*SHARED_SIZE_LIMIT*/), cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blockCount) *
                                             sycl::range<3>(1, 1, threadCount),
                                         sycl::range<3>(1, 1, threadCount)),
                       [=](sycl::nd_item<3> item_ct1) {
                         oddEvenMergeSortShared(d_DstKey, d_DstVal, d_SrcKey,
                                                d_SrcVal, SHARED_SIZE_LIMIT,
                                                dir, item_ct1,
                                                s_key_acc_ct1.get_pointer(),
                                                s_val_acc_ct1.get_pointer());
                       });
    });

    for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
      for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
        // Unlike with bitonic sort, combining bitonic merge steps with
        // stride = [SHARED_SIZE_LIMIT / 2 .. 1] seems to be impossible as there
        // are dependencies between data elements crossing the SHARED_SIZE_LIMIT
        // borders
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (batchSize * arrayLength) / 512) *
                    sycl::range<3>(1, 1, 256),
                sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
              oddEvenMergeGlobal(d_DstKey, d_DstVal, d_DstKey, d_DstVal,
                                 arrayLength, size, stride, dir, item_ct1);
            });
      }
  }
  return threadCount;
}
