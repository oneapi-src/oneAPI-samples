#ifndef __SNAPPY_READER_HPP__
#define __SNAPPY_READER_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "byte_stream.hpp"
#include "constexpr_math.hpp"         // included from ../../../../include
#include "metaprogramming_utils.hpp"  // included from ../../../../include

//
// Streams in bytes from InPipe 'literals_per_cycle' at a time and
// generates LZ77InputData (see ../common/common.hpp) to the OutPipe for the
// LZ77Decoder kernel.
//
//  Template parameters:
//    InPipe: a SYCL pipe that streams in compressed Snappy data,
//      'literals_per_cycle' bytes at a time.
//    OutPipe: a SYCL pipe that streams out either an array of literals with
//      a valid count (when reading a literal string) or a {length, distance}
//      pair (when doing a copy), in the form of LZ77InputData data.
//      This is the input the LZ77 decoder.
//    literals_per_cycle: the maximum number of literals read from the input
//      (and written to the output) at once.
//
//  Arguments:
//    in_count: the number of compressed bytes
//
template <typename InPipe, typename OutPipe, unsigned literals_per_cycle>
unsigned SnappyReader(unsigned in_count) {
  // ensure the InPipe and OutPipe are SYCL pipes
  static_assert(fpga_tools::is_sycl_pipe_v<InPipe>);
  static_assert(fpga_tools::is_sycl_pipe_v<OutPipe>);

  // the input and output pipe data types
  using InPipeBundleT = decltype(InPipe::read());
  using OutPipeBundleT = decltype(OutPipe::read());

  // make sure the input and output types are correct
  static_assert(std::is_same_v<InPipeBundleT, ByteSet<literals_per_cycle>>);
  static_assert(
      std::is_same_v<OutPipeBundleT,
                     FlagBundle<SnappyLZ77InputData<literals_per_cycle>>>);

  // the number of bits to count to 'literals_per_cycle'
  constexpr unsigned literals_per_cycle_bits =
      fpga_tools::Log2(literals_per_cycle) + 1;

  // the maximum number of bytes to read at once is max(literals_per_cycle, 5),
  // cases:
  //    - reading the preamble length is 1...5 bytes
  //    - reading a literal length can be 1...5 bytes
  //    - reading a copy command can be 1...5 bytes
  //    - reading literals can be 1...literals_per_cycle bytes
  constexpr unsigned kMaxReadBytes = fpga_tools::Max(literals_per_cycle, 5U);

  // the stream size should be double the maximum bytes we will need on each
  // iteration so that we always have bytes ready
  // the maximum number of bytes we need is the maximum of 5 and
  // literals_per_cycle
  constexpr unsigned kByteStreamSize = kMaxReadBytes * 2;

  // the byte stream
  using ByteStreamT = ByteStream<kByteStreamSize, kMaxReadBytes>;
  ByteStreamT byte_stream;

  unsigned data_read_in_preamble = 0;

  // the first 1...5 bytes indicate the number of bytes in the stream
  bool reading_preamble = true;
  unsigned preamble_count_local = 0;
  ac_uint<3> bytes_processed_in_preamble;

  // NOTE: this loop is expected to have a trip count of ~1-5 iterations and
  // therefore is not a performance critical loop. However, the compiler doesn't
  // know that and tries to optimize for throughput (~Fmax/II). We don't want
  // this loop to be our Fmax bottleneck, so increase the II.
  [[intel::initiation_interval(3)]]  // NO-FORMAT: Attribute
  while (reading_preamble) {
    if (byte_stream.Space() >= literals_per_cycle) {
      bool valid_read;
      auto pipe_data = InPipe::read(valid_read);
      if (valid_read) {
        byte_stream.template Write(pipe_data);
        data_read_in_preamble += literals_per_cycle;
      }
    }

    if (byte_stream.Count() >= 5) {
      // grab the 5 bytes
      auto first_five_bytes = byte_stream.template Read<5>();

      // the uncompressed length is in the range [0, 2^32) and is encoded with
      // a varint between 1 to 5 bytes. the top bit of each byte indicates
      // whether to keep reading, and the bottom seven bits are data.
      // For example, a length of 64 is encoded with 0x40, and a length
      // of 2097150 (0x1FFFFE) would be stored as 0xFE 0xFF 0x7F.
      // Below, we are grabbing the "keep going" bit (the MSB) and the data
      // bits (the 7 LSBs).
      ac_uint<1> first_five_bytes_use_bits[5];
      ac_uint<7> first_five_bytes_data[5];
#pragma unroll
      for (int i = 0; i < 5; i++) {
        auto b = first_five_bytes.byte[i];
        first_five_bytes_use_bits[i] = (b >> 7) & 0x1;
        first_five_bytes_data[i] = b & 0x7F;
      }

      // Now, we build the 5 possible uncompressed lengths assuming we use
      // 1 to 5 of the bytes
      unsigned preamble_counts[5];
#pragma unroll
      for (int i = 0; i < 5; i++) {
        preamble_counts[i] = 0;
#pragma unroll
        for (int j = 0; j < i + 1; j++) {
          preamble_counts[i] |= first_five_bytes_data[j].to_uint() << (j * 7);
        }
      }

      // now select the actual uncompressed length by checking the
      // "keep going" bit of each byte
      if (first_five_bytes_use_bits[0] == 0) {
        bytes_processed_in_preamble = 1;
        preamble_count_local = preamble_counts[0];
      } else if (first_five_bytes_use_bits[1] == 0) {
        bytes_processed_in_preamble = 2;
        preamble_count_local = preamble_counts[1];
      } else if (first_five_bytes_use_bits[2] == 0) {
        bytes_processed_in_preamble = 3;
        preamble_count_local = preamble_counts[2];
      } else if (first_five_bytes_use_bits[3] == 0) {
        bytes_processed_in_preamble = 4;
        preamble_count_local = preamble_counts[3];
      } else {
        bytes_processed_in_preamble = 5;
        preamble_count_local = preamble_counts[4];
      }

      // shift the byte stream by however many we used and flag that we
      // are done reading the preamble
      byte_stream.Shift(bytes_processed_in_preamble);
      reading_preamble = false;
    }
  }

  // are we reading a literal and how many more literals do we have to read
  bool reading_literal = false;
  unsigned literal_len_counter;

  unsigned data_read = data_read_in_preamble;
  bool all_data_read = data_read >= in_count;
  bool all_data_read_next = data_read >= (in_count - literals_per_cycle);


  // keep track of the number of bytes processed
  constexpr unsigned max_bytes_processed_inc =
      fpga_tools::Max((unsigned)5, literals_per_cycle);
  unsigned bytes_processed_next[max_bytes_processed_inc + 1];
  bool bytes_processed_in_range = true;
  bool bytes_processed_in_range_next[max_bytes_processed_inc + 1];
#pragma unroll
  for (int i = 0; i < max_bytes_processed_inc + 1; i++) {
    bytes_processed_next[i] = bytes_processed_in_preamble + i;
    bytes_processed_in_range_next[i] =
        bytes_processed_in_preamble + i < in_count;
  }

  // the output data
  bool out_ready = false;
  SnappyLZ77InputData<literals_per_cycle> out_data;

  // main processing loop
  // keep going while there is input to read, or data to read from byte_stream
  while (bytes_processed_in_range) {
    // grab new bytes if there is space
    if (byte_stream.Space() >= literals_per_cycle) {

      bool valid_read;
      auto pipe_data = InPipe::read(valid_read);
      if (valid_read) {
        byte_stream.template Write(pipe_data);
        data_read += literals_per_cycle;
        all_data_read = all_data_read_next;
        all_data_read_next = data_read >= (in_count - literals_per_cycle);

      }
    }

    if (!reading_literal) {
      // finding the next command, which is either a literal string
      // or copy command. We will need at most 5 bytes to get the command
      // in a single iteration, so make sure we have enough bytes to do so
      if (byte_stream.Count() >= 5 || all_data_read) {
        // grab the next 5 bytes
        auto five_bytes = byte_stream.template Read<5>();

        // what type of command is this, literal or copy?
        ac_uint<8> first_byte(five_bytes.byte[0]);
        ac_uint<2> first_byte_type = first_byte.slc<2>(0);
        ac_uint<6> first_byte_data = first_byte.slc<6>(2);

        //////////////////////////////////////////////////////////////////////
        // Assuming the command is a literal length
        // find all possible literal lengths, which could require 0 to 4 more
        // bytes
        ac_uint<6> literal_len_without_extra_bytes = first_byte_data;

        // if the len >= 60, then the number of extra bytes to read for the
        // length = len - 60 + 1 (1 to 4 bytes). So grab the low 2 bits of the
        // data from the first byte that, if the literal len >= 60, will be
        // the number of extra bytes to read minus 1 (to save on the number
        // of bits to store it, we will add 1 later).
        ac_uint<2> literal_len_extra_bytes = first_byte_data.slc<2>(0);

        // find all the possible literal lengths assuming we need to read 1 to
        // 4 more bytes
        unsigned literal_len_with_bytes[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          literal_len_with_bytes[i] = 0;
#pragma unroll
          for (int j = 0; j < i + 1; j++) {
            literal_len_with_bytes[i] |= (unsigned)(five_bytes.byte[j + 1])
                                         << (j * 8);
          }
        }
        //////////////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////
        // Assuming the command is a copy
        // find all possible copy lengths and offset combinations
        // a type 1 copy stores len-4 in the bits 4...2 of the first byte
        ac_uint<4> copy_len_type1 = first_byte_data.slc<3>(0) + ac_uint<3>(4);

        // a type 2 and 3 copy store len-1 in the top 6 bits of the first byte
        ac_uint<7> copy_len_type2_and_type3 = first_byte_data + ac_uint<1>(1);

        // a type 1 copy uses an 11 bit offset, with the high high 3 bits
        // being the bits 7...5 of the first byte, along with the next byte
        unsigned copy_offset_extra_bits_type1 =
            first_byte_data.slc<3>(3).to_uint();
        unsigned copy_offset_type1 = (copy_offset_extra_bits_type1 << 8) |
                                     (unsigned)(five_bytes.byte[1]);

        // type 2 and 3 copies use 16-bit and 32-bit offsets, respectively.
        // they are little-endian integers using the next 2 or 4 bytes,
        // respectively
        unsigned copy_offset_type2 = (unsigned)(five_bytes.byte[2]) << 8 |
                                     (unsigned)(five_bytes.byte[1]);
        unsigned copy_offset_type3 = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          copy_offset_type3 |= (unsigned)(five_bytes.byte[5 - i - 1])
                               << (8 * i);
        }
        //////////////////////////////////////////////////////////////////////

        // we have now built up every combination of commands for literals
        // and copies. Now it is time to figure out what we are actually doing
        ac_uint<3> bytes_used;
        if (first_byte_type == 0) {
          // LITERAL
          // calculate the literal length
          if (literal_len_without_extra_bytes < 60) {
            literal_len_counter =
                literal_len_without_extra_bytes + ac_uint<1>(1);
            bytes_used = 1;
          } else {
            literal_len_counter =
                literal_len_with_bytes[literal_len_extra_bytes] + 1;
            // extra bytes = literal_len_extra_bytes + 1, + 1 for first byte
            bytes_used = literal_len_extra_bytes + ac_uint<2>(2);
          }

          // NOTE: could grab the extra bytes and start writing literals
          // right away, but that may cost Fmax/II
          out_ready = false;
          reading_literal = true;
        } else if (first_byte_type == 1) {
          // COPY: with 1 extra byte for offset
          out_data.is_literal = false;
          out_data.length = copy_len_type1;
          out_data.distance = copy_offset_type1;
          bytes_used = 2;
          out_ready = true;
        } else if (first_byte_type == 2) {
          // COPY: with 2 extra bytes for offset
          out_data.is_literal = false;
          out_data.length = copy_len_type2_and_type3;
          out_data.distance = copy_offset_type2;
          bytes_used = 3;
          out_ready = true;
        } else {  // first_byte_type == 3
          // COPY: with 4 extra bytes for offset
          out_data.is_literal = false;
          out_data.length = copy_len_type2_and_type3;
          out_data.distance = copy_offset_type3;
          bytes_used = 5;
          out_ready = true;
        }

        // shift by however many bytes we used
        byte_stream.Shift(bytes_used);

        auto bytes_processed_next_val = bytes_processed_next[bytes_used];
        bytes_processed_in_range = bytes_processed_in_range_next[bytes_used];
#pragma unroll
        for (int i = 0; i < max_bytes_processed_inc + 1; i++) {
          bytes_processed_next[i] = bytes_processed_next_val + i;
          bytes_processed_in_range_next[i] =
              bytes_processed_next_val < in_count - i;
        }
      }
    } else {
      // reading a string of literals so figure out how many literals to read
      // in this iteration (in range [1, literals_per_cycle]) and whether we
      // should keep reading literals next iteration (if we do end up reading
      // the literals this iteration)
      ac_uint<literals_per_cycle_bits> amount_to_read;
      bool still_reading_literal;
      if (literal_len_counter < literals_per_cycle) {
        amount_to_read = literal_len_counter;
        still_reading_literal = false;
      } else {
        amount_to_read = literals_per_cycle;
        still_reading_literal = (literal_len_counter != literals_per_cycle);
      }

      // reading literals from input stream
      if (byte_stream.Count() >= amount_to_read) {
        // figure out how many literals will be valid
        // we can always subtract by 'literals_per_cycle' since this will only
        // go negative on the last iteration, which we detect with
        // 'still_reading_literal'
        literal_len_counter -= literals_per_cycle;

        // whether to keep reading the literals
        reading_literal = still_reading_literal;

        // read the literals (we know we have enough)
        auto literals = byte_stream.template Read<literals_per_cycle>();

        // build the output data
        out_data.is_literal = true;
        out_data.valid_count = amount_to_read;
#pragma unroll
        for (int i = 0; i < literals_per_cycle; i++) {
          out_data.literal[i] = literals.byte[i];
        }
        out_ready = true;

        // shift the byte stream by however many (valid) literals we wrote
        byte_stream.Shift(amount_to_read);

        auto bytes_processed_next_val = bytes_processed_next[amount_to_read];
        bytes_processed_in_range =
            bytes_processed_in_range_next[amount_to_read];
#pragma unroll
        for (int i = 0; i < max_bytes_processed_inc + 1; i++) {
          bytes_processed_next[i] = bytes_processed_next_val + i;
          bytes_processed_in_range_next[i] =
              bytes_processed_next_val < in_count - i;
        }
      }
    }

    // write the output
    if (out_ready) {
      OutPipe::write(OutPipeBundleT(out_data));
      out_ready = false;
    }
  }

  // notify downstream that we are done
  OutPipe::write(OutPipeBundleT(true));

  // return the preamble count
  return preamble_count_local;
}

template <typename Id, typename InPipe, typename OutPipe,
          unsigned literals_per_cycle>
sycl::event SubmitSnappyReader(sycl::queue& q, unsigned in_count,
                               unsigned* preamble_count_ptr) {
  return q.single_task<Id>([=] {
#if defined (IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer
    // lives on the device.
    // Knowing this, the compiler won't generate hardware to
    // potentially get data from the host.
    sycl::device_ptr<unsigned> preamble_count(preamble_count_ptr);
#else
    // Device pointers are not supported when targeting an FPGA 
    // family/part
    unsigned* preamble_count(preamble_count_ptr);
#endif
    *preamble_count =
        SnappyReader<InPipe, OutPipe, literals_per_cycle>(in_count);
  });
}

#endif /* __SNAPPY_READER_HPP__ */