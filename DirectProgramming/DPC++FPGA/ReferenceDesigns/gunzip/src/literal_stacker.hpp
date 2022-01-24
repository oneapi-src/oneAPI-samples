#ifndef __LITERAL_STACKER_HPP__
#define __LITERAL_STACKER_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

using namespace sycl;

template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
void LiteralStacker() {
  using OutPipeBundleT = FlagBundle<LiteralPack<literals_per_cycle>>;
  constexpr int cache_idx_bits = fpga_tools::Log2(literals_per_cycle*2) + 1;

  bool done;

  // cache up to literals_per_cycle * 2 elements so that we can always
  // write out literals_per_cycle valid elements in a row (except on the last
  // iteration)
  ac_uint<cache_idx_bits> cache_idx = 0;
  [[intel::fpga_register]] unsigned char cache_buf[literals_per_cycle * 2];

  do {
    // try to read in some data
    bool data_valid;
    auto pipe_data = InPipe::read(data_valid);
    done = pipe_data.flag && data_valid;

    if (data_valid && !done) {
      // add the valid data we read in to the cache
      #pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        if (i < pipe_data.data.valid_count) {
          cache_buf[cache_idx + i] = pipe_data.data.literal[i];
        }
      }
      cache_idx += pipe_data.data.valid_count;
    }

    // if there are enough elements in the cache to write out
    // 'literals_per_cycle' valid elements, or if the upstream kernel indicated
    // that it is done producing data, then write to the output pipe
    if (cache_idx >= literals_per_cycle || done) {
      // create the output pack of characters from the current cache
      LiteralPack<literals_per_cycle> out_pack;
      #pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        // copy the character
        out_pack.literal[i] = cache_buf[i];

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
      // it is safe to always subtract literals_per_cycle since that will only
      // happen on the last iteration of the outer while loop (when 'done'
      // is true)
      cache_idx -= ac_uint<cache_idx_bits>(literals_per_cycle);

      // write output
      OutPipe::write(OutPipeBundleT(out_pack));
    }
  } while (!done);
  
  // notify downstream kernel that we are done
  OutPipe::write(OutPipeBundleT(true));
}

template<typename Id, typename InPipe, typename OutPipe,
         unsigned literals_per_cycle>
event SubmitLiteralStacker(queue& q) {
  return q.single_task<Id>([=] {
    LiteralStacker<InPipe, OutPipe, literals_per_cycle>();
  });
}

#endif /* __LITERAL_STACKER_HPP__ */