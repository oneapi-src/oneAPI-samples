#ifndef __BYTE_STACKER_HPP__
#define __BYTE_STACKER_HPP__

// clang-format off
#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Included from DirectProgramming/DPC++FPGA/include/
#include "constexpr_math.hpp"
#include "metaprogramming_utils.hpp"
// clang-format on

//
// Data streams in the 'InPipe' pipe and can have between 0 to
// 'literals_per_cycle' valid elements per cycle. This function takes the input
// and "stacks" it, such that the output is always 'literals_per_cycle' valid
// elements (except possibly the last write).
//
//  Template parameters:
//    InPipe: a SYCL pipe that streams in an array of bytes and a `valid_count`,
//      which is in the range [0, literals_per_cycle]
//    OutPipe: a SYCL pipe that streams out an array of 'literals_per_cycle'
//      valid bytes on every write, except possibly the last iteration.
//    literals_per_cycle: the maximum valid bytes on the input and the number
//      valid bytes on the output (except possibly the last iteration).
//
template <typename InPipe, typename OutPipe, unsigned literals_per_cycle>
void ByteStacker() {
  // check that the input and output pipe types are actually pipes
  static_assert(fpga_tools::is_sycl_pipe_v<InPipe>);
  static_assert(fpga_tools::is_sycl_pipe_v<OutPipe>);

  // literals_per_cycle must be greater than 0
  static_assert(literals_per_cycle > 0);

  // input type rules:
  //  must have a member named 'flag' which is a boolean
  //  must have a member named 'data' which has a subscript operator and a
  //  member named 'valid_count'
  using InPipeBundleT = decltype(InPipe::read());
  static_assert(has_flag_bool_v<InPipeBundleT>);
  static_assert(has_data_member_v<InPipeBundleT>);
  using InDataT = decltype(std::declval<InPipeBundleT>().data);
  static_assert(fpga_tools::has_subscript_v<InDataT>);
  static_assert(has_valid_count_member_v<InDataT>);

  // output type rules:
  //  same as input data
  using OutPipeBundleT = decltype(OutPipe::read());
  static_assert(has_flag_bool_v<OutPipeBundleT>);
  static_assert(has_data_member_v<OutPipeBundleT>);
  using OutDataT = decltype(std::declval<OutPipeBundleT>().data);
  static_assert(fpga_tools::has_subscript_v<OutDataT>);
  static_assert(has_valid_count_member_v<OutDataT>);

  // the number of bits needed to count from 0 to 
  constexpr int cache_idx_bits = fpga_tools::Log2(literals_per_cycle * 2) + 1;

  // cache up to literals_per_cycle * 2 elements so that we can always
  // write out literals_per_cycle valid elements in a row (except on the last
  // iteration)
  ac_uint<cache_idx_bits> cache_idx = 0;
  [[intel::fpga_register]] unsigned char cache_buf[literals_per_cycle * 2];

  bool done = false;
  while (!done) {
    // try to read in some data
    bool data_valid;
    auto pipe_data = InPipe::read(data_valid);
    done = pipe_data.flag && data_valid;

    // add the valid data we read in to the cache
    if (data_valid && !done) {
#pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        if (i < pipe_data.data.valid_count) {
          cache_buf[cache_idx + i] = pipe_data.data.byte[i];
        }
      }
      cache_idx += pipe_data.data.valid_count;
    }

    // if there are enough elements in the cache to write out
    // 'literals_per_cycle' valid elements, or if the upstream kernel indicated
    // that it is done producing data, then write to the output pipe
    if (cache_idx >= literals_per_cycle || done) {
      // create the output pack of characters from the current cache
      BytePack<literals_per_cycle> out_pack;

#pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        // copy the character
        out_pack.byte[i] = cache_buf[i];

        // shift the extra characters to the front of the cache
        cache_buf[i] = cache_buf[i + literals_per_cycle];
      }

      // mark output with the number of valid elements
      if (cache_idx <= literals_per_cycle) {
        out_pack.valid_count = cache_idx;
      } else {
        out_pack.valid_count = literals_per_cycle;
      }

      // decrement cache_idx by number of elements we read
      // it is safe to always subtract literals_per_cycle since that can only
      // result in a negative number on the last iteration of the outer while
      // loop (when 'done' is true), at which point the value will never be used
      cache_idx -= ac_uint<cache_idx_bits>(literals_per_cycle);

      // write output
      OutPipe::write(OutPipeBundleT(out_pack));
    }
  }

  // notify downstream kernel that we are done
  OutPipe::write(OutPipeBundleT(true));
}

// Creates a kernel from the byte stacker kernel
template <typename Id, typename InPipe, typename OutPipe,
          unsigned literals_per_cycle>
sycl::event SubmitByteStacker(sycl::queue& q) {
  return q.single_task<Id>([=] {
    ByteStacker<InPipe, OutPipe, literals_per_cycle>();
  });
}

#endif /* __BYTE_STACKER_HPP__ */