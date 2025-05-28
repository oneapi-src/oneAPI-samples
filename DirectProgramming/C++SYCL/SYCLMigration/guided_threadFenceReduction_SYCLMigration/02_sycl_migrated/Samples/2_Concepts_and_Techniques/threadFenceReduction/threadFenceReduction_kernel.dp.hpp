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

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces
   the overall cost of the algorithm while keeping the work complexity O(n) and
   the step complexity O(log n). (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/

template <unsigned int blockSize>
void reduceBlock(volatile float *sdata, float mySum, const unsigned int tid,
                 sycl::group<3> cta, const sycl::nd_item<3> &item_ct1) {
  sycl::sub_group tile32 = item_ct1.get_sub_group();
  sdata[tid] = mySum;
  //item_ct1.get_sub_group().barrier();

  const int VEC = 32;
  const int vid = tid & (VEC - 1);

  float beta = mySum;
  float temp;

  for (int i = VEC / 2; i > 0; i >>= 1) {
    if (vid < i) {
      temp = sdata[tid + i];
      beta += temp;
      sdata[tid] = beta;
    }
//    item_ct1.get_sub_group().barrier();
  }
  item_ct1.barrier();

  if (item_ct1.get_local_linear_id() == 0) {
    beta = 0;
    for (int i = 0; i < item_ct1.get_local_range(2); i += VEC) {
      beta += sdata[i];
    }
    sdata[0] = beta;
  }
  item_ct1.barrier();
}

template <unsigned int blockSize, bool nIsPow2>
void reduceBlocks(const float *g_idata, float *g_odata, unsigned int n,
                  sycl::group<3> cta, const sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
  auto sdata = (float *)dpct_local;

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = item_ct1.get_local_id(2);
  unsigned int i =
      item_ct1.get_group(2) * (blockSize * 2) + item_ct1.get_local_id(2);
  unsigned int gridSize = blockSize * 2 * item_ct1.get_group_range(2);
  float mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  // do reduction in shared mem
  reduceBlock<blockSize>(sdata, mySum, tid, cta, item_ct1);

  // write result for this block to global mem
  if (tid == 0) g_odata[item_ct1.get_group(2)] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
void reduceMultiPass(const float *g_idata, float *g_odata,
                                unsigned int n,
                                const sycl::nd_item<3> &item_ct1,
                                uint8_t *dpct_local) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
  reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta, item_ct1,
                                   dpct_local);
}

// Global variable used by reduceSinglePass to count how many blocks have
// finished
dpct::global_memory<unsigned int, 0> retirementCount(0);

dpct::err0 setRetirementCount(int retCnt) try {
  return DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(retirementCount.get_ptr(), &retCnt, sizeof(unsigned int))
          .wait());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// This reduction kernel reduces an arbitrary size array in a single kernel
// invocation It does so by keeping track of how many blocks have finished.
// After each thread block completes the reduction of its own block of data, it
// "takes a ticket" by atomically incrementing a global counter.  If the ticket
// value is equal to the number of thread blocks, then the block holding the
// ticket knows that it is the last block to finish.  This last block is
// responsible for summing the results of all the other blocks.
//
// In order for this to work, we must be sure that before a block takes a
// ticket, all of its memory transactions have completed.  This is what
// __threadfence() does -- it blocks until the results of all outstanding memory
// transactions within the calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass
// approach), see the "reduction" sample in the CUDA SDK.
template <unsigned int blockSize, bool nIsPow2>
void reduceSinglePass(const float *g_idata, float *g_odata,
                                 unsigned int n,
                                 const sycl::nd_item<3> &item_ct1,
                                 uint8_t *dpct_local,
                                 unsigned int &retirementCount, bool &amLast) {
  // Handle to thread block group
  sycl::group<3> cta = item_ct1.get_group();
  //
  // PHASE 1: Process all inputs assigned to this block
  //

  reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta, item_ct1,
                                   dpct_local);

  //
  // PHASE 2: Last block finished will process all partial sums
  //

  if (item_ct1.get_group_range(2) > 1) {
    const unsigned int tid = item_ct1.get_local_id(2);

    auto smem = (float *)dpct_local;

    // wait until all outstanding memory instructions in this thread are
    // finished
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
   // __syncthreads();

    // Thread 0 takes a ticket
    if (tid == 0) {
      unsigned int ticket = dpct::atomic_fetch_compare_inc<
          sycl::access::address_space::generic_space>(
          &retirementCount, item_ct1.get_group_range(2));
      // If the ticket ID is equal to the number of blocks, we are the last
      // block!
      amLast = (ticket == item_ct1.get_group_range(2) - 1);
    }
    item_ct1.barrier();

    // The last block sums the results of all other blocks
    if (amLast) {
      int i = tid;
      float mySum = 0;

      while (i < item_ct1.get_group_range(2)) {
        mySum += g_odata[i];
        i += blockSize;
      }

      reduceBlock<blockSize>(smem, mySum, tid, cta, item_ct1);

      if (tid == 0) {
        g_odata[0] = smem[0];

        // reset retirement count so that next run succeeds
        retirementCount = 0;
      }
    }
  }
}

bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
extern "C" void reduce(int size, int threads, int blocks, float *d_idata,
                       float *d_odata) {
  sycl::range<3> dimBlock(1, 1, threads);
  sycl::range<3> dimGrid(1, 1, blocks);
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size)) {
    switch (threads) {
      case 512:
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<512, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 256:
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<256, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 128:

      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<128, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 64:
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<64, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 32:

      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<32, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 16:
    
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<16, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 8:
    
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<8, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 4:
    
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<4, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 2:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<2, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 1:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<1, true>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;
    }
  } else {
    switch (threads) {
      case 512:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<512, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 256:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<256, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 128:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<128, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 64:
        
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<64, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 32:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<32, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 16:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<16, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 8:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<8, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 4:
       
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<4, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 2:
        
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<2, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;

      case 1:
        
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(smemSize), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              reduceMultiPass<1, false>(
                  d_idata, d_odata, size, item_ct1,
                  dpct_local_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        break;
    }
  }
}

extern "C" void reduceSinglePass(int size, int threads, int blocks,
                                 float *d_idata, float *d_odata) {
  sycl::range<3> dimBlock(1, 1, threads);
  sycl::range<3> dimGrid(1, 1, blocks);
 
  int smemSize = threads * sizeof(float);

  // choose which of the optimized versions of reduction to launch
  if (isPow2(size)) {
    switch (threads) {
      case 512:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<512, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 256:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<256, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 128:
      
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<128, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 64:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<64, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 32:
        
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<32, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 16:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<16, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 8:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<8, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 4:
        
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<4, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 2:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<2, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 1:
     
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<1, true>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;
    }
  } else {
    switch (threads) {
      case 512:
        
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<512, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 256:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<256, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 128:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<128, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 64:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<64, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 32:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<32, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 16:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<16, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 8:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<8, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 4:
       
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<4, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 2:
        
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<2, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;

      case 1:
        
      {
        retirementCount.init();

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          auto retirementCount_ptr_ct1 = retirementCount.get_ptr();

          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smemSize), cgh);
          sycl::local_accessor<bool, 0> amLast_acc_ct1(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<3> item_ct1)
                  [[sycl::reqd_sub_group_size(32)]] {
                    reduceSinglePass<1, false>(
                        d_idata, d_odata, size, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                        *retirementCount_ptr_ct1, amLast_acc_ct1);
                  });
        });
      }
        break;
    }
  }
}

#endif  // #ifndef _REDUCE_KERNEL_H_
