#ifndef __LZ77_DECODER_HPP__
#define __LZ77_DECODER_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "common.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;

template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
void LZ77Decoder() {
  static_assert(literals_per_cycle > 0);
  static_assert(fpga_tools::IsPow2(literals_per_cycle));
  using OutPipeBundleT = FlagBundle<LiteralPack<literals_per_cycle>>;

  bool done;
  bool reading_history = false;
  bool reading_history_next;
  short history_counter;

  constexpr unsigned kMaxHistory = 32768;
  constexpr unsigned history_buffer_count = kMaxHistory / literals_per_cycle;
  constexpr unsigned history_buffer_buffer_idx_bits = fpga_tools::Log2(literals_per_cycle);
  constexpr unsigned history_buffer_buffer_idx_mask = literals_per_cycle - 1;
  constexpr unsigned history_buffer_idx_bits = fpga_tools::Log2(history_buffer_count);
  constexpr unsigned history_buffer_idx_mask = history_buffer_count - 1;
  
  using HistBufBufIdxT = ac_uint<history_buffer_buffer_idx_bits>;
  using HistBufIdxT = ac_uint<history_buffer_idx_bits>;

  // the history buffers
  HistBufBufIdxT history_buffer_buffer_idx = 0;
  [[intel::fpga_register]] HistBufIdxT history_buffer_idx[literals_per_cycle];
  fpga_tools::NTuple<unsigned char[history_buffer_count], literals_per_cycle> history_buffer;

  // the history buffer caches to cache in-flight writes and break loop carried
  // dependencies
  constexpr int kCacheDepth = 7;
  [[intel::fpga_register]] unsigned char history_buffer_cache_val[literals_per_cycle][kCacheDepth + 1];
  [[intel::fpga_register]] HistBufIdxT history_buffer_cache_idx[literals_per_cycle][kCacheDepth + 1];

  // the variables used to read from the history buffer upon request from the
  // Huffman decoder kernel 
  HistBufBufIdxT read_history_buffer_buffer_idx = 0;
  [[intel::fpga_register]] HistBufIdxT read_history_buffer_idx[literals_per_cycle];
  [[intel::fpga_register]] HistBufBufIdxT read_history_shuffle_idx[literals_per_cycle];

  // precompute the function
  //   dist_small + ((i - dist_small) % dist_small)
  constexpr auto mod_lut_init = [&] {
    constexpr int dim = literals_per_cycle - 1;
    std::array<unsigned char, dim * dim> ret{};
    for (int y = 0; y < dim; y++) {
      for (int x = y; x < dim; x++) {
        unsigned char dist = y + 1;
        unsigned char i = x + 1;
        ret[y * dim + x] = dist + ((i - dist) % dist);
      }
    }
    return ret;
  }();
  
  // ac_ints cannot be constexpr, so initialize an array of ac_int with
  // the mod_lut_init ROM that was computed above
  [[intel::fpga_register]] HistBufBufIdxT mod_lut[literals_per_cycle - 1][literals_per_cycle - 1];
  for (int y = 0; y < (literals_per_cycle - 1); y++) {
    for (int x = y; x < (literals_per_cycle - 1); x++) {
      mod_lut[y][x] = mod_lut_init[y * (literals_per_cycle - 1) + x];
    }
  }

  // initialize the index pointers for each history buffer
  #pragma unroll
  for (int i = 0; i < literals_per_cycle; i++) { history_buffer_idx[i] = 0; }

  [[intel::ivdep(kCacheDepth)]]
  do {
    bool data_valid = true;
    LiteralPack<literals_per_cycle> out_data;

    // if we aren't currently reading from the history, read from input pipe
    if (!reading_history) {
      // read from pipe
      auto pipe_data = InPipe::read(data_valid);

      // check if we are done
      done = pipe_data.flag && data_valid;

      // grab the symbol or the length and distance pair
      short len_or_sym = pipe_data.data.len_or_sym;
      short dist = pipe_data.data.dist_or_flag;

      out_data.literal[0] = len_or_sym & 0xFF;
      out_data.valid_count = 1;

      // if we get a length distance pair we will read 'len_or_sym' bytes
      // starting 'dist'
      history_counter = len_or_sym;
      reading_history = (dist > 0) && data_valid;
      reading_history_next = history_counter > literals_per_cycle;

      // grab the low Log2(literals_per_cycle) bits of the distance
      HistBufBufIdxT dist_small = dist & history_buffer_buffer_idx_mask;
      
      // find the first buffer we will read from given the distance
      read_history_buffer_buffer_idx = (history_buffer_buffer_idx - dist_small) & history_buffer_buffer_idx_mask;

      // find the starting read index for each buffer, as well as compute
      // the shuffle vector for shuffling the data to the output
      #pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        // the buffer index
        HistBufBufIdxT buf_idx = (read_history_buffer_buffer_idx + HistBufBufIdxT(i)) & history_buffer_buffer_idx_mask;

        // compute the starting index for buffer 'buf_idx' (in range
        // [0, literals_per_cycle))
        HistBufIdxT starting_read_idx_for_this_buf = (history_buffer_idx[buf_idx] - ((dist - i) / literals_per_cycle)) - 1;
        if (buf_idx == history_buffer_buffer_idx) {
          starting_read_idx_for_this_buf += 1;
        }
        read_history_buffer_idx[buf_idx] = starting_read_idx_for_this_buf & history_buffer_idx_mask;

        if (dist > i) {
          read_history_shuffle_idx[i] = buf_idx;
        } else {
          // this special case happens whenever dist < literals_per_cycle
          // and we need to repeat one of the earlier elements
          // idx_back = dist_small + ((i - dist_small) % dist_small);
          HistBufBufIdxT idx_back = mod_lut[dist_small - 1][i - 1];
          read_history_shuffle_idx[i] = (history_buffer_buffer_idx - idx_back) & history_buffer_buffer_idx_mask;
        }
      }
    }

    if (reading_history) {
      // grab from each of the history buffers
      unsigned char historical_bytes[literals_per_cycle];

      fpga_tools::UnrolledLoop<literals_per_cycle>([&](auto i) {
        auto idx_in_buf = read_history_buffer_idx[i];
        historical_bytes[i] = history_buffer.template get<i>()[idx_in_buf];

        // also check the cache to see if it is there
        #pragma unroll
        for (int j = 0; j < kCacheDepth + 1; j++) {
          if (history_buffer_cache_idx[i][j] == idx_in_buf) {
            historical_bytes[i] = history_buffer_cache_val[i][j];
          }
        }
      });

      // shuffle the elements read from the history to the output
      #pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        out_data.literal[i] = historical_bytes[read_history_shuffle_idx[i]];
      }

      if (history_counter < literals_per_cycle) {
        out_data.valid_count = history_counter;
      } else {
        out_data.valid_count = literals_per_cycle;
      }

      // update the history read index
      #pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        read_history_buffer_idx[i] = (read_history_buffer_idx[i] + ac_uint<1>(1)) & history_buffer_idx_mask;
      }

      // update whether we are still reading the history
      reading_history = reading_history_next;
      reading_history_next = history_counter > literals_per_cycle * 2;
      history_counter -= literals_per_cycle;
    }
    
    if (!done && data_valid) {
      // compute the valid bitmap for the writes
      bool write_bitmap[literals_per_cycle];
      [[intel::fpga_register]] HistBufBufIdxT shuffle_vec[literals_per_cycle];
      
      #pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        HistBufBufIdxT buf_idx = (history_buffer_buffer_idx + i) & history_buffer_buffer_idx_mask;
        write_bitmap[buf_idx] = i < out_data.valid_count;
        shuffle_vec[buf_idx] = i;
      }

      // write to the history buffers
      fpga_tools::UnrolledLoop<literals_per_cycle>([&](auto i) {
        if (write_bitmap[i]) {
          // grab the literal to write out
          auto literal_out = out_data.literal[shuffle_vec[i]];

          // the index into the buffer
          HistBufIdxT idx_in_buf = history_buffer_idx[i];
          history_buffer.template get<i>()[idx_in_buf] = literal_out;

          // write new value into cache as well
          history_buffer_cache_val[i][kCacheDepth] = literal_out;
          history_buffer_cache_idx[i][kCacheDepth] = idx_in_buf;

          // Cache is just a shift register, so shift the shift reg. Pushing
          // into the back of the shift reg is done above.
          #pragma unroll
          for (int j = 0; j < kCacheDepth; j++) {
            history_buffer_cache_val[i][j] = history_buffer_cache_val[i][j + 1];
            history_buffer_cache_idx[i][j] = history_buffer_cache_idx[i][j + 1];
          }

          // update the history buffer index
          history_buffer_idx[i] = (history_buffer_idx[i] + ac_uint<1>(1)) & history_buffer_idx_mask;
        }
      });

      // update the most recent buffer
      history_buffer_buffer_idx = (history_buffer_buffer_idx + out_data.valid_count) & history_buffer_buffer_idx_mask;

      // write the output to the pipe
      OutPipe::write(OutPipeBundleT(out_data));
    }
  } while (!done);
  
  OutPipe::write(OutPipeBundleT(true));
}

// special case of LZ77 decoder for 1 element per cycle
template<typename InPipe, typename OutPipe>
void LZ77Decoder() {
  using OutPipeBundleT = FlagBundle<LiteralPack<1>>;

  bool done;
  bool reading_history = false;
  bool reading_history_next;
  short history_counter;

  constexpr unsigned kMaxHistory = 32768;
  constexpr unsigned history_buffer_count = kMaxHistory;
  constexpr unsigned history_buffer_idx_bits = fpga_tools::Log2(history_buffer_count);
  constexpr unsigned history_buffer_idx_mask = history_buffer_count - 1;
  
  using HistBufIdxT = ac_uint<history_buffer_idx_bits>;

  // the history buffers
  HistBufIdxT history_buffer_idx = 0, read_history_buffer_idx;
  unsigned char history_buffer [history_buffer_count];

  // the history buffer caches to cache in-flight writes and break loop carried
  // dependencies
  constexpr int kCacheDepth = 7;
  [[intel::fpga_register]] unsigned char history_buffer_cache_val[kCacheDepth + 1];
  [[intel::fpga_register]] HistBufIdxT history_buffer_cache_idx[kCacheDepth + 1];

  [[intel::ivdep(kCacheDepth)]]
  do {
    bool data_valid = true;
    LiteralPack<1> out_data;

    // if we aren't currently reading from the history, read from input pipe
    if (!reading_history) {
      // read from pipe
      auto pipe_data = InPipe::read(data_valid);

      // check if we are done
      done = pipe_data.flag && data_valid;

      // grab the symbol or the length and distance pair
      short len_or_sym = pipe_data.data.len_or_sym;
      short dist = pipe_data.data.dist_or_flag;

      out_data.literal[0] = len_or_sym & 0xFF;
      out_data.valid_count = 1;

      // if we get a length distance pair we will read 'len_or_sym' bytes
      // starting 'dist'
      history_counter = len_or_sym;
      reading_history = (dist > 0) && data_valid;
      reading_history_next = history_counter > 1;

      // initialize the read index
      read_history_buffer_idx = (history_buffer_idx - dist) & history_buffer_idx_mask;
    }

    if (reading_history) {
      // read from the history buffer
      out_data.literal[0] = history_buffer[read_history_buffer_idx];

      // also check the cache to see if it is there
      #pragma unroll
      for (int j = 0; j < kCacheDepth + 1; j++) {
        if (history_buffer_cache_idx[j] == read_history_buffer_idx) {
          out_data.literal[0] = history_buffer_cache_val[j];
        }
      }
      out_data.valid_count = 1;

      // update the history read index
      read_history_buffer_idx =
          (read_history_buffer_idx + ac_uint<1>(1)) & history_buffer_idx_mask;

      // update whether we are still reading the history
      reading_history = reading_history_next;
      reading_history_next = history_counter > 2;
      history_counter--;
    }
    
    if (!done && data_valid) {
      // write to the most history buffer
      history_buffer[history_buffer_idx] = out_data.literal[0];

      // also add the most recent written value to the cache
      history_buffer_cache_val[kCacheDepth] = out_data.literal[0];
      history_buffer_cache_idx[kCacheDepth] = history_buffer_idx;
      #pragma unroll
      for (int j = 0; j < kCacheDepth; j++) {
        history_buffer_cache_val[j] = history_buffer_cache_val[j + 1];
        history_buffer_cache_idx[j] = history_buffer_cache_idx[j + 1];
      }

      // move the write index
      history_buffer_idx =
          (history_buffer_idx + ac_uint<1>(1)) & history_buffer_idx_mask;

      // write the output to the pipe
      OutPipe::write(OutPipeBundleT(out_data));
    }
  } while (!done);
  
  OutPipe::write(OutPipeBundleT(true));
}

template<typename Id, typename InPipe, typename OutPipe,
         unsigned literals_per_cycle>
event SubmitLZ77Decoder(queue& q) {
  return q.single_task<Id>([=] {
    if constexpr (literals_per_cycle == 1) {
      return LZ77Decoder<InPipe, OutPipe>();
    } else {
      return LZ77Decoder<InPipe, OutPipe, literals_per_cycle>();
    }
  });
}

#endif /* __LZ77_DECODER_HPP__ */