#ifndef __SORTINGNETWORKS_HPP__
#define __SORTINGNETWORKS_HPP__

#include <algorithm>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Included from DirectProgramming/C++SYCL_FPGA/include/
#include "constexpr_math.hpp"

using namespace sycl;

//
// Creates a merge sort network.
// Takes in two sorted lists ('a' and 'b') of size 'k_width' and merges them
// into a single sorted output in a single cycle, in the steady state.
//
// Convention:
//    a = {data[0], data[2], data[4], ...}
//    b = {data[1], data[3], data[5], ...}
//
template <typename ValueT, unsigned char k_width, class CompareFunc>
void MergeSortNetwork(sycl::vec<ValueT, k_width * 2>& data,
                      CompareFunc compare) {
  if constexpr (k_width == 4) {
    // Special case for k_width==4 that has 1 less compare on the critical path
    #pragma unroll
    for (unsigned char i = 0; i < 4; i++) {
      if (!compare(data[2 * i], data[2 * i + 1])) {
        std::swap(data[2 * i], data[2 * i + 1]);
      }
    }

    if (!compare(data[1], data[4])) {
      std::swap(data[1], data[4]);
    }
    if (!compare(data[3], data[6])) {
      std::swap(data[3], data[6]);
    }

    #pragma unroll
    for (unsigned char i = 0; i < 3; i++) {
      if (!compare(data[2 * i + 1], data[2 * i + 2])) {
        std::swap(data[2 * i + 1], data[2 * i + 2]);
      }
    }
  } else {
    // the general case
    // this works well for k_width = 1 or 2, but is not optimal for
    // k_width = 4 (see if-case above) or higher
    constexpr unsigned char merge_tree_depth = fpga_tools::Log2(k_width * 2);
    #pragma unroll
    for (unsigned i = 0; i < merge_tree_depth; i++) {
      #pragma unroll
      for (unsigned j = 0; j < k_width - i; j++) {
        if (!compare(data[i + 2 * j], data[i + 2 * j + 1])) {
          std::swap(data[i + 2 * j], data[i + 2 * j + 1]);
        }
      }
    }
  }
}

//
// Creates a bitonic sorting network.
// It accepts and sorts 'k_width' elements per cycle, in the steady state.
// For more info see: https://en.wikipedia.org/wiki/Bitonic_sorter
//
template <typename ValueT, unsigned char k_width, class CompareFunc>
void BitonicSortNetwork(sycl::vec<ValueT, k_width>& data, CompareFunc compare) {
  #pragma unroll
  for (unsigned char k = 2; k <= k_width; k *= 2) {
    #pragma unroll
    for (unsigned char j = k / 2; j > 0; j /= 2) {
      #pragma unroll
      for (unsigned char i = 0; i < k_width; i++) {
        const unsigned char l = i ^ j;
        if (l > i) {
          const bool comp = compare(data[i], data[l]);
          const bool cond1 = ((i & k) == 0) && !comp;
          const bool cond2 = ((i & k) != 0) && comp;
          if (cond1 || cond2) {
            std::swap(data[i], data[l]);
          }
        }
      }
    }
  }
}

//
// The sorting network kernel.
// This kernel streams in 'k_width' elements per cycle, sends them through a
// 'k_width' wide bitonic sorting network, and writes the sorted output or size
// 'k_width' to device memory. The result is an output array ('out_ptr') of size
// 'total_count' where each set of 'k_width' elements is sorted.
//
template <typename Id, typename ValueT, typename IndexT, typename InPipe,
          unsigned char k_width, class CompareFunc>
event SortNetworkKernel(queue& q, ValueT* out_ptr, IndexT total_count,
                        CompareFunc compare) {
  // the number of loop iterations required to process all of the data
  const IndexT iterations = total_count / k_width;

  return q.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
    // Creating a device_ptr tells the compiler that this pointer is in
    // device memory, not host memory, and avoids creating extra connections
    // to host memory
    // This is only done in the case where we target a BSP as device 
    // pointers are not supported when targeting an FPGA family/part
#if defined(IS_BSP)
    device_ptr<ValueT> out(out_ptr);
#else
    ValueT* out(out_ptr);
#endif

    for (IndexT i = 0; i < iterations; i++) {
      // read the input data from the pipe
      sycl::vec<ValueT, k_width> data = InPipe::read();

      // bitonic sort network sorts the k_width elements of 'data' in-place
      // NOTE: there are no dependencies across loop iterations on 'data'
      // here, so this sorting network can be fully pipelined
      BitonicSortNetwork<ValueT, k_width>(data, compare);

      // write the 'k_width' sorted elements to device memory
      #pragma unroll
      for (unsigned char j = 0; j < k_width; j++) {
        out[i * k_width + j] = data[j];
      }
    }
  });
}

#endif /* __SORTINGNETWORKS_HPP__ */
