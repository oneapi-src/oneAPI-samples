#ifndef __LZ77_DECODER_HPP__
#define __LZ77_DECODER_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "common.hpp"
#include "constexpr_math.hpp"            // included from ../../../../include
#include "metaprogramming_utils.hpp"     // included from ../../../../include
#include "onchip_memory_with_cache.hpp"  // included from ../../../../include
#include "tuple.hpp"                     // included from ../../../../include
#include "unrolled_loop.hpp"             // included from ../../../../include

//
// Performs LZ77 decoding for more than 1 element at once.
// Streams in 'LZ77InputData' (see common.hpp) appended with a flag (FlagBundle)
// which contains either a literal from upstream, or a {length, distance} pair.
// Given a literal input, this function simply tracks that literal in a history
// buffer and writes it to the output. For a {length, distance} pair, this
// function reads 'literals_per_cycle' elements from the history buffer per
// cycle and writes them to the output.
//
//  Template parameters:
//    InPipe: a SYCL pipe that streams in LZ77InputData with a boolean flag that
//      indicates whether the input stream is done.
//    OutPipe: a SYCL pipe that streams out an array of literals and a
//      'valid_count' that is in the range [0, literals_per_cycle].
//    literals_per_cycle: the number of literals to read from the history
//      buffer at once. This sets the maximum possible throughput for the
//      LZ77 decoder.
//    max_distance: the maximum distancefor a {length, distance} pair
//      For example, for DEFLATE this is 32K and for snappy this is 64K.
//
template <typename InPipe, typename OutPipe, size_t literals_per_cycle,
          size_t max_distance>
void LZ77DecoderMultiElement() {
  using OutPipeBundleT = decltype(OutPipe::read());
  using OutDataT = decltype(std::declval<OutPipeBundleT>().data);

  // we will cyclically partition the history to 'literals_per_cycle' buffers,
  // so each buffer gets this many elements
  constexpr size_t history_buffer_count = max_distance / literals_per_cycle;

  // number of bits to count from 0 to literals_per_cycle-1
  constexpr size_t history_buffer_buffer_idx_bits =
      fpga_tools::Log2(literals_per_cycle);

  // bit mask for counting from 0 to literals_per_cycle-1
  constexpr size_t history_buffer_buffer_idx_mask = literals_per_cycle - 1;

  // number of bits to count from 0 to history_buffer_count-1
  constexpr size_t history_buffer_idx_bits =
      fpga_tools::Log2(history_buffer_count);

  // bit mask for counting from 0 to history_buffer_count-1
  constexpr size_t history_buffer_idx_mask = history_buffer_count - 1;

  // the data type used to index from 0 to literals_per_cycle-1 (i.e., pick
  // which buffer to use)
  using HistBufBufIdxT = ac_uint<history_buffer_buffer_idx_bits>;

  // the data type used to index from 0 to history_buffer_count-1 (i.e., after
  // picking which buffer, index into that buffer)
  using HistBufIdxT = ac_uint<history_buffer_idx_bits>;

  // track whether we are reading from the history, and how many more elements
  // to read from the history
  bool reading_history = false;
  bool reading_history_next;
  short history_counter;

  // which of the 'literals_per_cycle' buffers is the one to write to next
  HistBufBufIdxT history_buffer_buffer_idx = 0;

  // for each of the 'literals_per_cycle' buffers, where do we write next
  [[intel::fpga_register]] HistBufIdxT history_buffer_idx[literals_per_cycle];


  // the OnchipMemoryWithCache history buffers cache in-flight writes to the 
  // history buffer and break loop carried dependencies that are smaller than
  // kCacheDepth
  constexpr int kCacheDepth = 8;
  // the history buffers
  fpga_tools::NTuple<fpga_tools::OnchipMemoryWithCache<
                         unsigned char, history_buffer_count, kCacheDepth + 1>,
                     literals_per_cycle>
      history_buffer;

  // these variables are used to read from the history buffer upon request from
  // the Huffman decoder kernel
  HistBufBufIdxT read_history_buffer_buffer_idx = 0;
  [[intel::fpga_register]]  // NO-FORMAT: Attribute
  HistBufIdxT read_history_buffer_idx[literals_per_cycle];
  [[intel::fpga_register]]  // NO-FORMAT: Attribute
  HistBufBufIdxT read_history_shuffle_idx[literals_per_cycle];

  // precompute the function: dist + ((i - dist) % dist)
  // which is used for the corner case when the copy distance is less than
  // 'literals_per_cycle'
  [[intel::fpga_register]]  // NO-FORMAT: Attribute
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
  for (int i = 0; i < literals_per_cycle; i++) {
    history_buffer_idx[i] = 0;
  }

  bool done = false;

  // the main processing loop.
  // Using the OnchipMemoryWithCache, we are able to break all loop carried
  // dependencies with a distance of 'kCacheDepth' and less
  while (!done) {
    bool data_valid = true;
    OutDataT out_data;

    if (!reading_history) {
      // if we aren't currently reading from the history buffers,
      // then read from input pipe
      auto pipe_data = InPipe::read(data_valid);

      // check if the upstream kernel is done sending us data
      done = pipe_data.flag && data_valid;

      // grab the literal or the length and distance pair
      unsigned short dist = pipe_data.data.distance;

      // for the case of literal(s), we will simply write it to the output
      // get the specific LZ77InputData type to see how many literals can come
      // in the input at once and that it is less than literals_per_cycle
      using InputLZ77DataT = decltype(pipe_data.data);
      static_assert(InputLZ77DataT::max_literals <= literals_per_cycle);
#pragma unroll
      for (int i = 0; i < InputLZ77DataT::max_literals; i++) {
        out_data[i] = pipe_data.data.literal[i];
      }
      out_data.valid_count = pipe_data.data.valid_count;

      // if we get a length distance pair we will read 'pipe_data.data.length'
      // bytes starting at and offset of 'dist'
      history_counter = pipe_data.data.length;
      reading_history = !pipe_data.data.is_literal && data_valid;
      reading_history_next = history_counter > literals_per_cycle;

      // grab the low Log2(literals_per_cycle) bits of the distance
      HistBufBufIdxT dist_small = dist & history_buffer_buffer_idx_mask;

      // find which of the history buffers we will read from first
      read_history_buffer_buffer_idx =
          (history_buffer_buffer_idx - dist_small) &
          history_buffer_buffer_idx_mask;

      // find the starting read index for each history buffer, and compute the
      // shuffle vector for shuffling the data to the output
#pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        // the buffer index
        HistBufBufIdxT buf_idx =
            (read_history_buffer_buffer_idx + HistBufBufIdxT(i)) &
            history_buffer_buffer_idx_mask;

        // compute the starting index for buffer 'buf_idx' (in range
        // [0, literals_per_cycle))
        HistBufIdxT starting_read_idx_for_this_buf =
            (history_buffer_idx[buf_idx] - ((dist - i) / literals_per_cycle)) -
            1;
        if (buf_idx == history_buffer_buffer_idx) {
          starting_read_idx_for_this_buf += 1;
        }
        read_history_buffer_idx[buf_idx] =
            starting_read_idx_for_this_buf & history_buffer_idx_mask;

        if (dist > i) {
          // normal case for ths shuffle vector
          read_history_shuffle_idx[i] = buf_idx;
        } else {
          // EDGE CASE!
          // this special case happens whenever dist < literals_per_cycle
          // and we need to repeat one of the earlier elements
          // idx_back = dist_small - ((i - dist_small) % dist_small));
          HistBufBufIdxT idx_back = mod_lut[dist_small - 1][i - 1];
          read_history_shuffle_idx[i] = (history_buffer_buffer_idx - idx_back) &
                                        history_buffer_buffer_idx_mask;
        }
      }
    }

    if (reading_history) {
      // grab from each of the history buffers
      unsigned char historical_bytes[literals_per_cycle];
      fpga_tools::UnrolledLoop<literals_per_cycle>([&](auto i) {
        // get the index into this buffer and read from it
        auto idx_in_buf = read_history_buffer_idx[i];
        historical_bytes[i] = history_buffer.template get<i>().read(idx_in_buf);
      });

      // shuffle the elements read from the history buffers to the output
      // using the shuffle vector computed earlier. Note, the numbers in the
      // shuffle vector need not be unique, which happens in the special case
      // of dist < literals_per_cycle, described above.
#pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        out_data[i] = historical_bytes[read_history_shuffle_idx[i]];
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
        read_history_buffer_idx[i] =
            (read_history_buffer_idx[i] + ac_uint<1>(1)) &
            history_buffer_idx_mask;
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
      [[intel::fpga_register]] HistBufBufIdxT shuffle_vec[literals_per_cycle];

#pragma unroll
      for (int i = 0; i < literals_per_cycle; i++) {
        HistBufBufIdxT buf_idx =
            (history_buffer_buffer_idx + i) & history_buffer_buffer_idx_mask;
        write_bitmap[buf_idx] = i < out_data.valid_count;
        shuffle_vec[buf_idx] = i;
      }

      // write to the history buffers
      fpga_tools::UnrolledLoop<literals_per_cycle>([&](auto i) {
        if (write_bitmap[i]) {
          // grab the literal to write out
          auto literal_out = out_data[shuffle_vec[i]];

          // the index into this history buffer
          HistBufIdxT idx_in_buf = history_buffer_idx[i];
          // history_buffer.template get<i>()[idx_in_buf] = literal_out;
          history_buffer.template get<i>().write(idx_in_buf, literal_out);

          // update the history buffer index
          history_buffer_idx[i] =
              (history_buffer_idx[i] + ac_uint<1>(1)) & history_buffer_idx_mask;
        }
      });

      // update the most recent buffer
      history_buffer_buffer_idx =
          (history_buffer_buffer_idx + out_data.valid_count) &
          history_buffer_buffer_idx_mask;

      // write the output to the pipe
      OutPipe::write(OutPipeBundleT(out_data));
    }
  }

  OutPipe::write(OutPipeBundleT(true));
}

//
// For performance reasons, we provide a special version of LZ77 for
// literals_per_cycle = 1. See the comments on the LZ77DecoderMultiElement
// function above for information on the template parameters.
//
template <typename InPipe, typename OutPipe, size_t max_distance>
void LZ77DecoderSingleElement() {
  using OutPipeBundleT = decltype(OutPipe::read());
  using OutDataT = decltype(std::declval<OutPipeBundleT>().data);

  constexpr size_t history_buffer_count = max_distance;
  constexpr size_t history_buffer_idx_bits =
      fpga_tools::Log2(history_buffer_count);
  constexpr size_t history_buffer_idx_mask = history_buffer_count - 1;

  using HistBufIdxT = ac_uint<history_buffer_idx_bits>;

  // track whether we are reading from the history, and how many more elements
  // to read from the history
  bool reading_history = false;
  bool reading_history_next;
  short history_counter;

  // the history buffers
  HistBufIdxT history_buffer_idx = 0, read_history_buffer_idx;
  unsigned char history_buffer[history_buffer_count];

  // the history buffer caches to cache in-flight writes and break loop carried
  // dependencies
  constexpr int kCacheDepth = 7;
  [[intel::fpga_register]]  // NO-FORMAT: Attribute
  unsigned char history_buffer_cache_val[kCacheDepth + 1];
  [[intel::fpga_register]]  // NO-FORMAT: Attribute
  HistBufIdxT history_buffer_cache_idx[kCacheDepth + 1];

  bool done = false;

  [[intel::ivdep(kCacheDepth)]]  // NO-FORMAT: Attribute
  while (!done) {
    bool data_valid = true;
    OutDataT out_data;

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
      out_data[0] = pipe_data.data.literal[0];
      out_data.valid_count = ac_uint<1>(1);

      // if we get a length distance pair we will read 'pipe_data.data.length'
      // bytes starting at and offset of 'dist'
      history_counter = pipe_data.data.length;
      reading_history = !pipe_data.data.is_literal && data_valid;
      reading_history_next = history_counter > 1;

      // initialize the read index
      read_history_buffer_idx =
          (history_buffer_idx - dist) & history_buffer_idx_mask;
    }

    if (reading_history) {
      // read from the history buffer
      out_data[0] = history_buffer[read_history_buffer_idx];

      // also check the cache to see if it is there
#pragma unroll
      for (int j = 0; j < kCacheDepth + 1; j++) {
        if (history_buffer_cache_idx[j] == read_history_buffer_idx) {
          out_data[0] = history_buffer_cache_val[j];
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
      history_buffer[history_buffer_idx] = out_data[0];

      // also add the most recent written value to the cache
      history_buffer_cache_val[kCacheDepth] = out_data[0];
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
  }

  OutPipe::write(OutPipeBundleT(true));
}

//
// The top level LZ77 decoder that selects between the single- and
// multi-element variants above, at compile time.
//
//  Template parameters:
//    InPipe: a SYCL pipe that streams in LZ77InputData with a boolean flag that
//      indicates whether the input stream is done.
//    OutPipe: a SYCL pipe that streams out an array of literals and a
//      'valid_count' that is in the range [0, literals_per_cycle].
//    literals_per_cycle: the number of literals to read from the history
//      buffer at once. This sets the maximum possible throughput for the
//      LZ77 decoder.
//    max_distance: the maximum distancefor a {length, distance} pair
//      For example, for DEFLATE this is 32K and for snappy this is 64K.
//    max_length: the maximum length for a {length, distance} pair.
//
template <typename InPipe, typename OutPipe, size_t literals_per_cycle,
          size_t max_distance, size_t max_length>
void LZ77Decoder() {
  // check that the input and output pipe types are actually pipes
  static_assert(fpga_tools::is_sycl_pipe_v<InPipe>);
  static_assert(fpga_tools::is_sycl_pipe_v<OutPipe>);

  // these numbers need to be greater than 0 and powers of 2
  static_assert(literals_per_cycle > 0);
  static_assert(max_distance > 0);
  static_assert(max_length > 0);
  static_assert(fpga_tools::IsPow2(literals_per_cycle));
  static_assert(fpga_tools::IsPow2(max_distance));

  // input type rules:
  //  must have a member named 'flag' which is a boolean
  //  must have a member named 'data' which is an instance of LZ77InputData
  //  the max_distance and max_length of LZ77InputData must match the function's
  using InPipeBundleT = decltype(InPipe::read());
  static_assert(has_flag_bool_v<InPipeBundleT>);
  static_assert(has_data_member_v<InPipeBundleT>);
  using InDataT = decltype(std::declval<InPipeBundleT>().data);
  static_assert(is_lz77_input_data_v<InDataT>);
  static_assert(InDataT::literals_per_cycle <= literals_per_cycle);
  static_assert(InDataT::max_distance == max_distance);
  static_assert(InDataT::max_length == max_length);

  // output type rules:
  //  must have a member named 'flag' which is a boolean
  //  must have a member named 'data' which has a subscript operator and a
  //  member named 'valid_count'
  using OutPipeBundleT = decltype(OutPipe::read());
  static_assert(has_flag_bool_v<OutPipeBundleT>);
  static_assert(has_data_member_v<OutPipeBundleT>);
  using OutDataT = decltype(std::declval<OutPipeBundleT>().data);
  static_assert(fpga_tools::has_subscript_v<OutDataT>);
  static_assert(has_valid_count_member_v<OutDataT>);

  // make sure we can construct the OutPipeBundleT from OutDataT and/or a bool
  static_assert(std::is_constructible_v<OutPipeBundleT, OutDataT>);
  static_assert(std::is_constructible_v<OutPipeBundleT, OutDataT, bool>);
  static_assert(std::is_constructible_v<OutPipeBundleT, bool>);

  // select which LZ77 decoder version to use based on literals_per_cycle
  // at compile time
  if constexpr (literals_per_cycle == 1) {
    return LZ77DecoderSingleElement<InPipe, OutPipe, max_distance>();
  } else {
    return LZ77DecoderMultiElement<InPipe, OutPipe, literals_per_cycle,
                                   max_distance>();
  }
}

//
// Creates a kernel from the LZ77 decoder function
//
template <typename Id, typename InPipe, typename OutPipe,
          size_t literals_per_cycle, size_t max_distance, size_t max_length>
sycl::event SubmitLZ77Decoder(sycl::queue& q) {
  return q.single_task<Id>([=] {
    return LZ77Decoder<InPipe, OutPipe, literals_per_cycle, max_distance,
                       max_length>();
  });
}

#endif /* __LZ77_DECODER_HPP__ */