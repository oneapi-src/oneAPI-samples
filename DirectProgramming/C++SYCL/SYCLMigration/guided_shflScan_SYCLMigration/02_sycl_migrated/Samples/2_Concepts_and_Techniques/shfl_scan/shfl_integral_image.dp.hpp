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

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// Utility function to extract unsigned chars from an
// unsigned integer
sycl::uchar4 uint_to_uchar4(const unsigned int in) {
  return sycl::uchar4((in & 0x000000ff) >> 0, (in & 0x0000ff00) >> 8,
                      (in & 0x00ff0000) >> 16, (in & 0xff000000) >> 24);
}

// Utility for dealing with vector data at different levels.
struct packed_result {
  sycl::uint4 x, y, z, w;
};

packed_result get_prefix_sum(const sycl::uint4 &data, const sycl::group<3> &cta,
                             const sycl::nd_item<3> &item_ct1,
                             unsigned int *sums) {
  const auto tile = item_ct1.get_sub_group();

  const unsigned int lane_id = item_ct1.get_sub_group().get_local_linear_id();
  const unsigned int warp_id = item_ct1.get_sub_group().get_group_linear_id();

  unsigned int result[16] = {};
  {
    const sycl::uchar4 a = uint_to_uchar4(data.x());
    const sycl::uchar4 b = uint_to_uchar4(data.y());
    const sycl::uchar4 c = uint_to_uchar4(data.z());
    const sycl::uchar4 d = uint_to_uchar4(data.w());

    result[0] = a.x();
    result[1] = a.x() + a.y();
    result[2] = a.x() + a.y() + a.z();
    result[3] = a.x() + a.y() + a.z() + a.w();

    result[4] = b.x();
    result[5] = b.x() + b.y();
    result[6] = b.x() + b.y() + b.z();
    result[7] = b.x() + b.y() + b.z() + b.w();

    result[8] = c.x();
    result[9] = c.x() + c.y();
    result[10] = c.x() + c.y() + c.z();
    result[11] = c.x() + c.y() + c.z() + c.w();

    result[12] = d.x();
    result[13] = d.x() + d.y();
    result[14] = d.x() + d.y() + d.z();
    result[15] = d.x() + d.y() + d.z() + d.w();
  }

#pragma unroll
  for (unsigned int i = 4; i <= 7; i++) result[i] += result[3];

#pragma unroll
  for (unsigned int i = 8; i <= 11; i++) result[i] += result[7];

#pragma unroll
  for (unsigned int i = 12; i <= 15; i++) result[i] += result[11];

  unsigned int sum = result[15];

  // the prefix sum for each thread's 16 value is computed,
  // now the final sums (result[15]) need to be shared
  // with the other threads and add.  To do this,
  // the __shfl_up() instruction is used and a shuffle scan
  // operation is performed to distribute the sums to the correct
  // threads

#pragma unroll
  for (unsigned int i = 1; i < 32; i *= 2) {
    const unsigned int n =
        sycl::shift_group_right(item_ct1.get_sub_group(), sum, i);

    if (lane_id >= i) {
#pragma unroll
      for (unsigned int j = 0; j < 16; j++) {
        result[j] += n;
      }

      sum += n;
    }
  }

  // Now the final sum for the warp must be shared
  // between warps.  This is done by each warp
  // having a thread store to shared memory, then
  // having some other warp load the values and
  // compute a prefix sum, again by using __shfl_up.
  // The results are uniformly added back to the warps.
  // last thread in the warp holding sum of the warp
  // places that in shared
  if (item_ct1.get_sub_group().get_local_linear_id() ==
      (item_ct1.get_sub_group().get_local_linear_range() - 1)) {
    sums[warp_id] = result[15];
  }

  item_ct1.barrier();

  if (warp_id == 0) {
    unsigned int warp_sum = sums[lane_id];

#pragma unroll
    for (unsigned int i = 1; i <= 16; i *= 2) {
      const unsigned int n =
          sycl::shift_group_right(item_ct1.get_sub_group(), warp_sum, i);

      if (lane_id >= i) warp_sum += n;
    }

    sums[lane_id] = warp_sum;
  }

  item_ct1.barrier();

  // fold in unused warp
  if (warp_id > 0) {
    const unsigned int blockSum = sums[warp_id - 1];

#pragma unroll
    for (unsigned int i = 0; i < 16; i++) {
      result[i] += blockSum;
    }
  }

  packed_result out;
  memcpy(&out, result, sizeof(out));
  return out;
}

// This function demonstrates some uses of the shuffle instruction
// in the generation of an integral image (also
// called a summed area table)
// The approach is two pass, a horizontal (scanline) then a vertical
// (column) pass.
// This is the horizontal pass kernel.
void shfl_intimage_rows(const sycl::uint4 *img, sycl::uint4 *integral_image,
                        const sycl::nd_item<3> &item_ct1, unsigned int *sums) {
  const auto cta = item_ct1.get_group();
  const auto tile = item_ct1.get_sub_group();

  const unsigned int id = item_ct1.get_local_id(2);
  // pointer to head of current scanline
  const sycl::uint4 *scanline = &img[item_ct1.get_group(2) * 120];
  packed_result result = get_prefix_sum(scanline[id], cta, item_ct1, sums);

  // This access helper allows packed_result to stay optimized as registers
  // rather than spill to stack
  auto idxToElem = [&result](unsigned int idx) -> const sycl::uint4 {
    switch (idx) {
      case 0:
        return result.x;
      case 1:
        return result.y;
      case 2:
        return result.z;
      case 3:
        return result.w;
    }
    return {};
  };

  // assemble result
  // Each thread has 16 values to write, which are
  // now integer data (to avoid overflow).  Instead of
  // each thread writing consecutive uint4s, the
  // approach shown here experiments using
  // the shuffle command to reformat the data
  // inside the registers so that each thread holds
  // consecutive data to be written so larger contiguous
  // segments can be assembled for writing.
  /*
    For example data that needs to be written as

    GMEM[16] <- x0 x1 x2 x3 y0 y1 y2 y3 z0 z1 z2 z3 w0 w1 w2 w3
    but is stored in registers (r0..r3), in four threads (0..3) as:

    threadId   0  1  2  3
      r0      x0 y0 z0 w0
      r1      x1 y1 z1 w1
      r2      x2 y2 z2 w2
      r3      x3 y3 z3 w3

      after apply __shfl_xor operations to move data between registers r1..r3:

    threadId  00 01 10 11
              x0 y0 z0 w0
     xor(01)->y1 x1 w1 z1
     xor(10)->z2 w2 x2 y2
     xor(11)->w3 z3 y3 x3

     and now x0..x3, and z0..z3 can be written out in order by all threads.

     In the current code, each register above is actually representing
     four integers to be written as uint4's to GMEM.
  */

  const unsigned int idMask = id & 3;
  const unsigned int idSwizzle = (id + 2) & 3;
  const unsigned int idShift = (id >> 2) << 4;
  const unsigned int blockOffset = item_ct1.get_group(2) * 480;

  // Use CG tile to warp shuffle vector types
  result.y = sycl::permute_group_by_xor(item_ct1.get_sub_group(), result.y, 1);
  result.z = sycl::permute_group_by_xor(item_ct1.get_sub_group(), result.z, 2);
  result.w = sycl::permute_group_by_xor(item_ct1.get_sub_group(), result.w, 3);

  // First batch
  integral_image[blockOffset + idMask + idShift] = idxToElem(idMask);
  // Second batch offset by 2
  integral_image[blockOffset + idSwizzle + idShift + 8] = idxToElem(idSwizzle);

  // continuing from the above example,
  // this use of __shfl_xor() places the y0..y3 and w0..w3 data
  // in order.
  result.x = sycl::permute_group_by_xor(item_ct1.get_sub_group(), result.x, 1);
  result.y = sycl::permute_group_by_xor(item_ct1.get_sub_group(), result.y, 1);
  result.z = sycl::permute_group_by_xor(item_ct1.get_sub_group(), result.z, 1);
  result.w = sycl::permute_group_by_xor(item_ct1.get_sub_group(), result.w, 1);

  // First batch
  integral_image[blockOffset + idMask + idShift + 4] = idxToElem(idMask);
  // Second batch offset by 2
  integral_image[blockOffset + idSwizzle + idShift + 12] = idxToElem(idSwizzle);
}

// This kernel computes columnwise prefix sums.  When the data input is
// the row sums from above, this completes the integral image.
// The approach here is to have each block compute a local set of sums.
// First , the data covered by the block is loaded into shared memory,
// then instead of performing a sum in shared memory using __syncthreads
// between stages, the data is reformatted so that the necessary sums
// occur inside warps and the shuffle scan operation is used.
// The final set of sums from the block is then propagated, with the block
// computing "down" the image and adding the running sum to the local
// block sums.
void shfl_vertical_shfl(unsigned int *img, int width, int height,
                        const sycl::nd_item<3> &item_ct1,
                        sycl::local_accessor<unsigned int, 2> sums) {

  int tidx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
  // int warp_id = threadIdx.x / warpSize ;
  unsigned int lane_id = tidx % 8;
  // int rows_per_thread = (height / blockDim. y) ;
  // int start_row = rows_per_thread * threadIdx.y;
  unsigned int stepSum = 0;
  unsigned int mask = 0xffffffff;

  sums[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)] = 0;
  
  item_ct1.barrier();

  for (int step = 0; step < 135; step++) {
    unsigned int sum = 0;
    unsigned int *p =
        img + (item_ct1.get_local_id(1) + step * 8) * width + tidx;

    sum = *p;
    sums[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)] = sum;
    
    item_ct1.barrier();

    // place into SMEM
    // shfl scan reduce the SMEM, reformating so the column
    // sums are computed in a warp
    // then read out properly
    int partial_sum = 0;
    int j = item_ct1.get_local_id(2) % 8;
    int k = item_ct1.get_local_id(2) / 8 + item_ct1.get_local_id(1) * 4;

    partial_sum = sums[k][j];

    for (int i = 1; i <= 8; i *= 2) {
      
      int n = dpct::experimental::shift_sub_group_right(
          mask, item_ct1.get_sub_group(), partial_sum, i);

      if (lane_id >= i) partial_sum += n;
    }

    sums[k][j] = partial_sum;
    
    item_ct1.barrier();

    if (item_ct1.get_local_id(1) > 0) {
      sum += sums[item_ct1.get_local_id(2)][item_ct1.get_local_id(1) - 1];
    }

    sum += stepSum;
    stepSum += sums[item_ct1.get_local_id(2)][item_ct1.get_local_range(1) - 1];
    
    item_ct1.barrier();
    *p = sum;
  }
}
