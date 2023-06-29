// Copyright (C) 2013-2023 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

/* This is the top-level device source file for the fft2d example. 
 *
 * A 2D FFT transform requires applying a 1D FFT transform to each matrix row
 * followed by a 1D FFT transform to each column of the intermediate result.
 * 
 * A single FFT engine can process rows and columns back-to-back. However, as 
 * matrix data is stored in global memory, the efficiency of memory accesses 
 * will impact the overall performance. Accessing consecutive memory 
 * locations leads to efficient access patterns. However, this is obviously 
 * not possible when accessing both rows and columns.
 * 
 * The implementation is divided between three concurrent OpenCL kernels, as 
 * depicted below:
 *
 *  ---------------------      ---------------      -------------------------
 *  | read matrix rows, | ---> |  FFT engine | ---> | bit-reverse, transpose |
 *  |  form 8 streams   |  x8  |    (task)   |  x8  |    and write matrix    |
 *  |    (ND range)     | ---> |             | ---> |      (ND range)        |
 *  ---------------------      ---------------      --------------------------
 *
 * This sequence of kernels does back-to-back row processing followed by a 
 * data transposition and writes the results back to memory. The host code 
 * runs these kernels twice to produce the overall 2D FFT transform
 *
 * The FFT engine is implemented as an OpenCL single work-item task (see 
 * fft1d example for details) while the data reordering kernels, reading and 
 * writing the matrix data from / to memory, are implemented as ND range 
 * kernels. 
 *
 * These kernels transfer data through channels, an Altera Vendor extension 
 * that allows for direct communication between kernels or between kernels. 
 * This avoids the need to read and write intermediate data using global 
 * memory as in traditional OpenCL implementations.
 *
 * In many cases the FFT engine is a building block in a large application. In
 * this case, the memory layout of the matrix can be altered to achieve higher
 * memory transfer efficiency. This implementation demonstrates how an 
 * alternative memory layout can improve performance. The host switches 
 * between the two memory layouts using a kernel argument. See the 
 * 'mangle_bits' function for additional details.
 */

// Include source code for an engine that produces 8 points each step
#include "fft_8.cl" 

#define LOGPOINTS 3
#define POINTS (1 << LOGPOINTS)

// Source the log(size) (log(1k) = 10) from a header shared with the host code
#include "../host/inc/fft_config.h"

// Declare channels for kernel to kernel communication

#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float2 chan0 __attribute__((depth(0)));
channel float2 chan1 __attribute__((depth(0)));
channel float2 chan2 __attribute__((depth(0)));
channel float2 chan3 __attribute__((depth(0)));

channel float2 chan4 __attribute__((depth(0)));
channel float2 chan5 __attribute__((depth(0)));
channel float2 chan6 __attribute__((depth(0)));
channel float2 chan7 __attribute__((depth(0)));

channel float2 chanin0 __attribute__((depth(0)));
channel float2 chanin1 __attribute__((depth(0)));
channel float2 chanin2 __attribute__((depth(0)));
channel float2 chanin3 __attribute__((depth(0)));

channel float2 chanin4 __attribute__((depth(0)));
channel float2 chanin5 __attribute__((depth(0)));
channel float2 chanin6 __attribute__((depth(0)));
channel float2 chanin7 __attribute__((depth(0)));

// This utility function bit-reverses an integer 'x' of width 'bits'.

int bit_reversed(int x, int bits) {
  int y = 0;
  #pragma unroll 
  for (int i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  return y;
}

/* Accesses to DDR memory are efficient if the memory locations are accessed
 * in order. There is significant overhead when accesses are not in order. 
 * The penalty is higher if the accesses stride a large number of locations.
 *
 * This function provides the mapping for an alternative memory layout. This 
 * layout preserves some amount of linearity when accessing elements from the 
 * same matrix row, while bringing closer locations from the same matrix
 * column. The matrix offsets are represented using 2 * log(N) bits. This
 * function swaps bits log(N) - 1 ... log(N) / 2 with bits 
 * log(N) + log(N) / 2 - 1 ... log(N).
 *
 * The end result is that 2^(N/2) locations from the same row would still be 
 * consecutive in memory, while the distance between locations from the same 
 * column would be only 2^(N/2)
 */

int mangle_bits(int x) {
   const int NB = LOGN / 2;
   int a95 = x & (((1 << NB) - 1) << NB);
   int a1410 = x & (((1 << NB) - 1) << (2 * NB));
   int mask = ((1 << (2 * NB)) - 1) << NB;
   a95 = a95 << NB;
   a1410 = a1410 >> NB;
   return (x & ~mask) | a95 | a1410;
}

/* This kernel reads the matrix data and provides 8 parallel streams to the 
 * FFT engine. Each workgroup reads 8 matrix rows to local memory. Once this 
 * data has been buffered, the workgroup produces 8 streams from strided 
 * locations in local memory, according to the requirements of the FFT engine.
 */

__attribute__((reqd_work_group_size((1 << LOGN), 1, 1)))
kernel void fetch(global float2 * restrict src, int mangle) {
  const int N = (1 << LOGN);

  // Local memory for storing 8 rows
  local float2 buf[8 * N];

  float2x8 data;

  // Each read fetches 8 matrix points
  int x = get_global_id(0) << LOGPOINTS;

  /* When using the alternative memory layout, each row consists of a set of
   * segments placed far apart in memory. Instead of reading all segments from
   * one row in order, read one segment from each row before switching to the
   *  next segment. This requires swapping bits log(N) + 2 ... log(N) with 
   *  bits log(N) / 2 + 2 ... log(N) / 2 in the offset. 
   */

  int inrow, incol, where, where_global;
  if (mangle) {
    const int NB = LOGN / 2;
    int a1210 = x & ((POINTS - 1) << (2 * NB));
    int a75 = x & ((POINTS - 1) << NB);
    int mask = ((POINTS - 1) << NB) | ((POINTS - 1) << (2 * NB));
    a1210 >>= NB;
    a75 <<= NB;
    where = (x & ~mask) | a1210 | a75;
    where_global = mangle_bits(where);
  } else {
    where = x;
    where_global = where;
  }

  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))] = src[where_global];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 1] = src[where_global + 1];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 2] = src[where_global + 2];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 3] = src[where_global + 3];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 4] = src[where_global + 4];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 5] = src[where_global + 5];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 6] = src[where_global + 6];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 7] = src[where_global + 7];

  barrier(CLK_LOCAL_MEM_FENCE);

  int row = get_local_id(0) >> (LOGN - LOGPOINTS);
  int col = get_local_id(0) & (N / POINTS - 1);

  // Stream fetched data over 8 channels to the FFT engine

  write_channel_intel(chanin0, buf[row * N + col]);
  write_channel_intel(chanin1, buf[row * N + 4 * N / 8 + col]);
  write_channel_intel(chanin2, buf[row * N + 2 * N / 8 + col]);
  write_channel_intel(chanin3, buf[row * N + 6 * N / 8 + col]);
  write_channel_intel(chanin4, buf[row * N + N / 8 + col]);
  write_channel_intel(chanin5, buf[row * N + 5 * N / 8 + col]);
  write_channel_intel(chanin6, buf[row * N + 3 * N / 8 + col]);
  write_channel_intel(chanin7, buf[row * N + 7 * N / 8 + col]);
}

/* This single work-item task wraps the FFT engine
 * 'inverse' toggles between the direct and the inverse transform
 */

kernel void fft2d(int inverse) {
  const int N = (1 << LOGN);

  /* The FFT engine requires a sliding window for data reordering; data stored
   * in this array is carried across loop iterations and shifted by 1 element
   * every iteration; all loop dependencies derived from the uses of this 
   * array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  // needs to run "N / 8 - 1" additional iterations to drain the last outputs
  for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
    float2x8 data;

    // Read data from channels
    if (i < N * (N / POINTS)) {
      data.i0 = read_channel_intel(chanin0);
      data.i1 = read_channel_intel(chanin1);
      data.i2 = read_channel_intel(chanin2);
      data.i3 = read_channel_intel(chanin3);
      data.i4 = read_channel_intel(chanin4);
      data.i5 = read_channel_intel(chanin5);
      data.i6 = read_channel_intel(chanin6);
      data.i7 = read_channel_intel(chanin7);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    // Perform one FFT step
    data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

    // Write result to channels
    if (i >= N / POINTS - 1) {
      write_channel_intel(chan0, data.i0);
      write_channel_intel(chan1, data.i1);
      write_channel_intel(chan2, data.i2);
      write_channel_intel(chan3, data.i3);
      write_channel_intel(chan4, data.i4);
      write_channel_intel(chan5, data.i5);
      write_channel_intel(chan6, data.i6);
      write_channel_intel(chan7, data.i7);
    }
  }
}

/* This kernel receives the FFT results, buffers 8 rows and then writes the
 * results transposed in memory. Because 8 rows are buffered, 8 consecutive
 * columns can be written at a time on each transposed row. This provides some
 * degree of locality. In addition, when using the alternative matrix format,
 * consecutive rows are closer in memory, and this is also beneficial for  
 * higher memory access efficiency
 */

__attribute__((reqd_work_group_size((1 << LOGN), 1, 1)))
kernel void transpose(global float2 * restrict dest, int mangle) {
  const int N = (1 << LOGN);
  local float2 buf[POINTS * N];
  buf[8 * get_local_id(0)] = read_channel_intel(chan0);
  buf[8 * get_local_id(0) + 1] = read_channel_intel(chan1);
  buf[8 * get_local_id(0) + 2] = read_channel_intel(chan2);
  buf[8 * get_local_id(0) + 3] = read_channel_intel(chan3);
  buf[8 * get_local_id(0) + 4] = read_channel_intel(chan4);
  buf[8 * get_local_id(0) + 5] = read_channel_intel(chan5);
  buf[8 * get_local_id(0) + 6] = read_channel_intel(chan6);
  buf[8 * get_local_id(0) + 7] = read_channel_intel(chan7);
 
  barrier(CLK_LOCAL_MEM_FENCE);
  int colt = get_local_id(0);
  int revcolt = bit_reversed(colt, LOGN);
  int i = get_global_id(0) >> LOGN;
  int where = colt * N + i * POINTS;
  if (mangle) where = mangle_bits(where);
  dest[where] = buf[revcolt];
  dest[where + 1] = buf[N + revcolt];
  dest[where + 2] = buf[2 * N + revcolt];
  dest[where + 3] = buf[3 * N + revcolt];
  dest[where + 4] = buf[4 * N + revcolt];
  dest[where + 5] = buf[5 * N + revcolt];
  dest[where + 6] = buf[6 * N + revcolt];
  dest[where + 7] = buf[7 * N + revcolt];
}

