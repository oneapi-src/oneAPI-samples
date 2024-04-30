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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <helper_cuda.h>
#include "histogram_common.h"

////////////////////////////////////////////////////////////////////////////////
// Shortcut shared memory atomic addition functions
////////////////////////////////////////////////////////////////////////////////

#define TAG_MASK 0xFFFFFFFFU
inline void addByte(uint *s_WarpHist, uint data, uint threadTag) {
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
      s_WarpHist + data, 1);
}

inline void addWord(uint *s_WarpHist, uint data, uint tag) {
  addByte(s_WarpHist, (data >> 0) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 8) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
  addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

void histogram256Kernel(uint *d_PartialHistograms, uint *d_Data,
                                   uint dataCount,
                                   const sycl::nd_item<3> &item_ct1,
                                   uint *s_Hist) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
  // Per-warp subhistogram storage

  uint *s_WarpHist = s_Hist + (item_ct1.get_local_id(2) >> LOG2_WARP_SIZE) *
                                  HISTOGRAM256_BIN_COUNT;

// Clear shared memory storage for current threadblock before processing
#pragma unroll

  for (uint i = 0;
       i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE);
       i++) {
    s_Hist[item_ct1.get_local_id(2) + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
  }

  // Cycle through the entire data set, update subhistograms for each warp
  const uint tag = item_ct1.get_local_id(2) << (UINT_BITS - LOG2_WARP_SIZE);

  item_ct1.barrier();

  for (uint pos = UMAD(item_ct1.get_group(2), item_ct1.get_local_range(2),
                       item_ct1.get_local_id(2));
       pos < dataCount;
       pos += UMUL(item_ct1.get_local_range(2), item_ct1.get_group_range(2))) {
    uint data = d_Data[pos];
    addWord(s_WarpHist, data, tag);
  }

  // Merge per-warp histograms into per-block and write to global memory
  item_ct1.barrier();

  for (uint bin = item_ct1.get_local_id(2); bin < HISTOGRAM256_BIN_COUNT;
       bin += HISTOGRAM256_THREADBLOCK_SIZE) {
    uint sum = 0;

    for (uint i = 0; i < WARP_COUNT; i++) {
      sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
    }

    d_PartialHistograms[item_ct1.get_group(2) * HISTOGRAM256_BIN_COUNT + bin] =
        sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

void mergeHistogram256Kernel(uint *d_Histogram,
                                        uint *d_PartialHistograms,
                                        uint histogramCount,
                                        const sycl::nd_item<3> &item_ct1,
                                        uint *data) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();

  uint sum = 0;

  for (uint i = item_ct1.get_local_id(2); i < histogramCount;
       i += MERGE_THREADBLOCK_SIZE) {
    sum +=
        d_PartialHistograms[item_ct1.get_group(2) + i * HISTOGRAM256_BIN_COUNT];
  }

  data[item_ct1.get_local_id(2)] = sum;

  for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    item_ct1.barrier();

    if (item_ct1.get_local_id(2) < stride) {
      data[item_ct1.get_local_id(2)] += data[item_ct1.get_local_id(2) + stride];
    }
  }

  if (item_ct1.get_local_id(2) == 0) {
    d_Histogram[item_ct1.get_group(2)] = data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////
// histogram256kernel() intermediate results buffer
static const uint PARTIAL_HISTOGRAM256_COUNT = 240;
static uint *d_PartialHistograms;

// Internal memory allocation
extern "C" void initHistogram256(sycl::queue &sycl_queue) {
  
  DPCT_CHECK_ERROR(d_PartialHistograms = sycl::malloc_device<uint>(
                           PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT,
                           sycl_queue));
}

// Internal memory deallocation
extern "C" void closeHistogram256(sycl::queue &sycl_queue) {
  DPCT_CHECK_ERROR(
      sycl::free(d_PartialHistograms, sycl_queue));
}

extern "C" void histogram256(uint *d_Histogram, void *d_Data, uint byteCount, sycl::queue &sycl_queue) {
  assert(byteCount % sizeof(uint) == 0);
  sycl_queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint, 1> s_Hist_acc_ct1(
        sycl::range<1>(1536 /*HISTOGRAM256_THREADBLOCK_MEMORY*/), cgh);

    uint *d_PartialHistograms_ct0 = d_PartialHistograms;
    uint byteCount_sizeof_uint_ct2 = byteCount / sizeof(uint);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, PARTIAL_HISTOGRAM256_COUNT) *
                sycl::range<3>(1, 1, HISTOGRAM256_THREADBLOCK_SIZE),
            sycl::range<3>(1, 1, HISTOGRAM256_THREADBLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          histogram256Kernel(
              d_PartialHistograms_ct0, (uint *)d_Data,
              byteCount_sizeof_uint_ct2, item_ct1,
              s_Hist_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });

  sycl_queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint, 1> data_acc_ct1(
        sycl::range<1>(256 /*MERGE_THREADBLOCK_SIZE*/), cgh);

    uint *d_PartialHistograms_ct1 = d_PartialHistograms;
    uint PARTIAL_HISTOGRAM256_COUNT_ct2 = PARTIAL_HISTOGRAM256_COUNT;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, HISTOGRAM256_BIN_COUNT) *
                              sycl::range<3>(1, 1, MERGE_THREADBLOCK_SIZE),
                          sycl::range<3>(1, 1, MERGE_THREADBLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          mergeHistogram256Kernel(
              d_Histogram, d_PartialHistograms_ct1,
              PARTIAL_HISTOGRAM256_COUNT_ct2, item_ct1,
              data_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });
}
