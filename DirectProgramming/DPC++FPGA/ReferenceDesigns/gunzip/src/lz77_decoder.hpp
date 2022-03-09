#ifndef __LZ77_DECODER_HPP__
#define __LZ77_DECODER_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "common.hpp"
#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;

//
// Performs LZ77 decoding for more than 1 element per cycle.
// Streams in a 'FlagBundle' containing a 'LZ77InputData' (see common.hpp)
// which contains either a literal from upstream, or a {length, distance} pair.
// Given a literal input, this function simply tracks that literal in a history
// buffer and writes it to the output. For a {length, distance} pair, this
// this function reads 'literals_per_cycle' elements from the history buffer
// per cycle and writes them to the output.
//
template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
void LZ77DecoderMultiElement() {
  static_assert(literals_per_cycle > 0);
  static_assert(fpga_tools::IsPow2(literals_per_cycle));
  using OutPipeBundleT = FlagBundle<BytePack<literals_per_cycle>>;

  bool done;
  bool reading_history = false;
  bool reading_history_next;
  short history_counter;

  // maximum history size is set based on the DEFLATE compression description
  constexpr unsigned kMaxHistory = 32768;

  // we will cyclically partition the history to 'literals_per_cycle' buffers,
  // so each buffer gets this many elements
  constexpr unsigned history_buffer_count = kMaxHistory / literals_per_cycle;

  // number of bits to count from 0 to literals_per_cycle-1
  constexpr unsigned history_buffer_buffer_idx_bits =
      fpga_tools::Log2(literals_per_cycle);

  // bit mask for counting from 0 to literals_per_cycle-1
  constexpr unsigned history_buffer_buffer_idx_mask = literals_per_cycle - 1;

  // number of bits to count from 0 to history_buffer_count-1
  constexpr unsigned history_buffer_idx_bits =
      fpga_tools::Log2(history_buffer_count);

  // bit mask for counting from 0 to history_buffer_count-1
  constexpr unsigned history_buffer_idx_mask = history_buffer_count - 1;
  
  // the data type used to index from 0 to literals_per_cycle-1 (i.e., pick
  // which buffer to use)
  using HistBufBufIdxT = ac_uint<history_buffer_buffer_idx_bits>;

  // the data type used to index from 0 to history_buffer_count-1 (i.e., after
  // picking which buffer, index into that buffer)
  using HistBufIdxT = ac_uint<history_buffer_idx_bits>;

  // which of the 'literals_per_cycle' buffers is the one to write to next
  HistBufBufIdxT history_buffer_buffer_idx = 0;

  // for each of the 'literals_per_cycle' buffers, where do we write next
  [[intel::fpga_register]]
  HistBufIdxT history_buffer_idx[literals_per_cycle];

  // the history buffers
  fpga_tools::NTuple<unsigned char[history_buffer_count], literals_per_cycle> history_buffer;

  // these shift-register caches cache in-flight writes to the history buffer
  // and break loop carried dependencies
  constexpr int kCacheDepth = 7;
  [[intel::fpga_register]]
  unsigned char history_buffer_cache_val[literals_per_cycle][kCacheDepth + 1];
  [[intel::fpga_register]]
  HistBufIdxT history_buffer_cache_idx[literals_per_cycle][kCacheDepth + 1];

  // these variables are used to read from the history buffer upon request from
  // the Huffman decoder kernel 
  HistBufBufIdxT read_history_buffer_buffer_idx = 0;
  [[intel::fpga_register]]
  HistBufIdxT read_history_buffer_idx[literals_per_cycle];
  [[intel::fpga_register]]
  HistBufBufIdxT read_history_shuffle_idx[literals_per_cycle];

  // precompute the function: dist + ((i - dist) % dist)
  // which is used for the special when the copy distance is less than
  // 'literals_per_cycle'
  [[intel::fpga_register]]
  constexpr auto mod_lut = [&] {
    constexpr int dim = literals_per_cycle - 1;
    std::array<std::array<unsigned char, dim>, dim> ret{};
    for (int y = 0; y < dim; y++) {
      for (int x = y; x < dim; x++) {
        unsigned char dist = y + 1;
        unsigned char i = x + 1;
        ret[y][x] = dist - ((i - dist) % dist);
      }
    }
    return ret;
  }();

  // initialize the index pointers for each history buffer
  #pragma unroll
  for (int i = 0; i < literals_per_cycle; i++) { history_buffer_idx[i] = 0; }

  // the main processing loop.
  // Using the shift-register cache, we are able to break all loop carried
  // dependencies with a distance of 'kCacheDepth' and less
  [[intel::ivdep(kCacheDepth)]]
  do {
    bool data_valid = true;
    BytePack<literals_per_cycle> out_data;

    if (!reading_history) {
      // if we aren't currently reading from the history buffers,
      // then read from input pipe
      auto pipe_data = InPipe::read(data_valid);

      // check if the upstream kernel is done sending us data
      done = pipe_data.flag && data_valid;

      // grab the literal or the length and distance pair
      unsigned short dist = pipe_data.data.distance;

      // for the case of literal(s), we will simply write it to the output
      #pragma unroll
      for (int i = 0; i < decltype(pipe_data.data)::max_symbols; i++) {
        out_data.byte[i] = pipe_data.data.literal[i];
      }
      out_data.valid_count = pipe_data.data.valid_count;

      // if we get a length distance pair we will read 'pipe_data.data.length'
      // bytes starting at and offset of 'dist'
      history_counter = pipe_data.data.length;
      reading_history = pipe_data.data.is_copy && data_valid;
      reading_history_next = history_counter > literals_per_cycle;

      // grab the low Log2(literals_per_cycle) bits of the distance
      HistBufBufIdxT dist_small = dist & history_buffer_buffer_idx_mask;
      
      // find which of the history buffers we will read from first
      read_history_buffer_buffer_idx = (history_buffer_buffer_idx - dist_small) & history_buffer_buffer_idx_mask;

      // find the starting read index for each history buffer, and compute the
      // shuffle vector for shuffling the data to the output
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
          // normal case for ths shuffle vector
          read_history_shuffle_idx[i] = buf_idx;
        } else {
          // this special case happens whenever dist < literals_per_cycle
          // and we need to repeat one of the earlier elements
          // idx_back = dist_small - ((i - dist_small) % dist_small));
          HistBufBufIdxT idx_back = mod_lut[dist_small - 1][i - 1];
          read_history_shuffle_idx[i] = (history_buffer_buffer_idx - idx_back) & history_buffer_buffer_idx_mask;
        }
      }
    }

    if (reading_history) {
      // grab from each of the history buffers
      unsigned char historical_bytes[literals_per_cycle];
      fpga_tools::UnrolledLoop<literals_per_cycle>([&](auto i) {
        // get the index into this buffer and read from it
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

      // shuffle the elements read from the history buffers to the output
      // using the shuffle vector computed earlier. Note, the numbers in the
      // shuffle vector need not be unique, which happens in the special case
      // of dist < literals_per_cycle, described above.
      #pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        out_data.byte[i] = historical_bytes[read_history_shuffle_idx[i]];
      }

      // we will write out min(history_counter, literals_per_cycle)
      // this can happen when the length of the copy is not a multiple of
      // 'literals_per_cycle'
      if (history_counter < literals_per_cycle) {
        out_data.valid_count = history_counter;
      } else {
        out_data.valid_count = literals_per_cycle;
      }

      // update the history read indices for the next iteration (if we are still
      // reading from the history buffers)
      #pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        read_history_buffer_idx[i] = (read_history_buffer_idx[i] + ac_uint<1>(1)) & history_buffer_idx_mask;
      }

      // update whether we will still be reading from the history buffers on
      // the next iteration of the loop
      reading_history = reading_history_next;
      reading_history_next = history_counter > literals_per_cycle * 2;
      history_counter -= literals_per_cycle;
    }
    
    if (!done && data_valid) {
      // compute the valid bitmap and shuffle vector for the writes
      bool write_bitmap[literals_per_cycle];
      [[intel::fpga_register]]
      HistBufBufIdxT shuffle_vec[literals_per_cycle];
      
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
          auto literal_out = out_data.byte[shuffle_vec[i]];

          // the index into this history buffer
          HistBufIdxT idx_in_buf = history_buffer_idx[i];
          history_buffer.template get<i>()[idx_in_buf] = literal_out;

          // write the new value into cache as well
          history_buffer_cache_val[i][kCacheDepth] = literal_out;
          history_buffer_cache_idx[i][kCacheDepth] = idx_in_buf;

          // the cache is just a shift register, so shift the shift reg. Pushing
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

//
// Same as above but for 1 element per cycle
//
template<typename InPipe, typename OutPipe>
void LZ77DecoderSingleElement() {
  using OutPipeBundleT = decltype(OutPipe::read());

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
  [[intel::fpga_register]]
  unsigned char history_buffer_cache_val[kCacheDepth + 1];
  [[intel::fpga_register]]
  HistBufIdxT history_buffer_cache_idx[kCacheDepth + 1];

  [[intel::ivdep(kCacheDepth)]]
  do {
    bool data_valid = true;
    BytePack<1> out_data;

    // if we aren't currently reading from the history, read from input pipe
    if (!reading_history) {
      // if we aren't currently reading from the history buffers,
      // then read from input pipe
      auto pipe_data = InPipe::read(data_valid);

      // check if the upstream kernel is done sending us data
      done = pipe_data.flag && data_valid;

      // grab the literal or the length and distance pair
      unsigned short dist = pipe_data.data.distance;

      // for the case of a literal, we will simply write it to the output
      out_data.byte[0] = pipe_data.data.literal[0];
      out_data.valid_count = ac_uint<1>(1);

      // if we get a length distance pair we will read 'pipe_data.data.length'
      // bytes starting at and offset of 'dist'
      history_counter = pipe_data.data.length;
      reading_history = pipe_data.data.is_copy && data_valid;
      reading_history_next = history_counter > 1;

      // initialize the read index
      read_history_buffer_idx = (history_buffer_idx - dist) & history_buffer_idx_mask;
    }

    if (reading_history) {
      // read from the history buffer
      out_data.byte[0] = history_buffer[read_history_buffer_idx];

      // also check the cache to see if it is there
      #pragma unroll
      for (int j = 0; j < kCacheDepth + 1; j++) {
        if (history_buffer_cache_idx[j] == read_history_buffer_idx) {
          out_data.byte[0] = history_buffer_cache_val[j];
        }
      }
      out_data.valid_count = ac_uint<1>(1);

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
      history_buffer[history_buffer_idx] = out_data.byte[0];

      // also add the most recent written value to the cache
      history_buffer_cache_val[kCacheDepth] = out_data.byte[0];
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

//
// Top-level LZ77 decoder to switch between the single- and multi-element per
// cycle variants above.
//
template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
void LZ77Decoder() {
  if constexpr (literals_per_cycle == 1) {
    return LZ77DecoderSingleElement<InPipe, OutPipe>();
  } else {
    return LZ77DecoderMultiElement<InPipe, OutPipe, literals_per_cycle>();
  }
}

//
// Creates a kernel from the LZ77 decoder function
//
template<typename Id, typename InPipe, typename OutPipe,
         unsigned literals_per_cycle>
event SubmitLZ77Decoder(queue& q) {
  return q.single_task<Id>([=] {
    return LZ77Decoder<InPipe, OutPipe, literals_per_cycle>();
  });
}

#endif /* __LZ77_DECODER_HPP__ */