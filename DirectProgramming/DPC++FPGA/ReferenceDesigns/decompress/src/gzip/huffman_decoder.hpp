#ifndef __HUFFMAN_DECODER_HPP__
#define __HUFFMAN_DECODER_HPP__

// clang-format off
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Included from DirectProgramming/DPC++FPGA/include/
#include "constexpr_math.hpp"
#include "metaprogramming_utils.hpp"

#include "byte_bit_stream.hpp"
#include "../common/common.hpp"
// clang-format on

// the size of the ByteBitStream buffer can be set from the compile comand.
// this number was set experimentally
#ifndef BIT_BUFFER_BITS
#define BIT_BUFFER_BITS 48
#endif
constexpr int kBitBufferBits = BIT_BUFFER_BITS;

// maximum bits to read dynamically from the bitstream
// more can be read statically using the templated Shift function
// in the main loop, we are reading 30 bits every iteration, but for reading
// in the first and second table, we read a dynamic number of bits, with a
// maximum of 5
constexpr int kBitBufferMaxReadBits = 5;

// maximum bits consumed in one shift of the bitstream
// the maximum number of bytes we will consume in one iteration of the main loop
// is 30 bits
constexpr int kBitBufferMaxShiftBits = 30;

static_assert(kBitBufferBits > 8);  // need to store at least a byte
static_assert(kBitBufferBits > kBitBufferMaxReadBits);
static_assert(kBitBufferBits > kBitBufferMaxShiftBits);

// the ByteBitStream alias
using BitStreamT = ByteBitStream<kBitBufferBits, kBitBufferMaxReadBits,
                                 kBitBufferMaxShiftBits>;

// constants for the static Huffman table
constexpr int kDeflateStaticNumLitLenCodes = 288;
constexpr int kDeflateStaticNumDistCodes = 32;
constexpr int kDeflateStaticTotalCodes =
    kDeflateStaticNumLitLenCodes + kDeflateStaticNumDistCodes;

// forward declare the helper functions that are defined at the end of the file
namespace detail {
template <typename InPipe>
std::pair<bool, ac_uint<2>> ParseLastBlockAndBlockType(BitStreamT& bit_stream);

template <typename InPipe>
void ParseFirstTable(BitStreamT& bit_stream,
                     ac_uint<9>& numlitlencodes,
                     ac_uint<6>& numdistcodes,
                     ac_uint<5>& numcodelencodes,
                     ac_uint<8> codelencode_map_first_code[8],
                     ac_uint<8> codelencode_map_last_code[8],
                     ac_uint<5> codelencode_map_base_idx[8],
                     ac_uint<5> codelencode_map[19]);
                    
template <typename InPipe>
void ParseSecondTable(BitStreamT& bit_stream,
                      bool is_static_huffman_block,
                      ac_uint<9> numlitlencodes,
                      ac_uint<6> numdistcodes,
                      ac_uint<5> numcodelencodes,
                      ac_uint<8> codelencode_map_first_code[8],
                      ac_uint<8> codelencode_map_last_code[8],
                      ac_uint<5> codelencode_map_base_idx[8],
                      ac_uint<5> codelencode_map[19],
                      ac_uint<15> lit_map_first_code[15],
                      ac_uint<15> lit_map_last_code[15],
                      ac_uint<9> lit_map_base_idx[15],
                      ac_uint<9> lit_map[286],
                      ac_uint<15> dist_map_first_code[15],
                      ac_uint<15> dist_map_last_code[15],
                      ac_uint<5> dist_map_base_idx[15],
                      ac_uint<5> dist_map[32]);

}  // namespace detail

//
// Performs Huffman Decoding.
// Streams in multiple DEFLATE blocks, 1 byte at a time, and performs huffman
// decoding. Supports uncompressed, statically compressed, and dynamically
// compressed blocks. For dynamically compressed blocks, the huffman tables are
// also built from the input byte stream.
//
template <typename InPipe, typename OutPipe>
void HuffmanDecoder() {
  // ensure the InPipe and OutPipe are SYCL pipes
  static_assert(fpga_tools::is_sycl_pipe_v<InPipe>);
  static_assert(fpga_tools::is_sycl_pipe_v<OutPipe>);
  
  using OutPipeBundleT = decltype(OutPipe::read());

  BitStreamT bit_stream;
  bool last_block = false;
  bool done_reading = false;

  // processing consecutive blocks
  // loop pipelining is disabled here because to reduce the amount of memory
  // utilization caused by replicating local variables. Since the inner
  // loop (the while(!block_done) loop) is the main processing loop, disabling
  // pipelining here does not have a significant affect on the throughput
  // of the design
  [[intel::disable_loop_pipelining]]  // NO-FORMAT: Attribute
  while (!last_block) {
    ////////////////////////////////////////////////////////////////////////////
    // BEGIN: parse the first three bits of the block
    auto [last_block_tmp, block_type] =
        detail::ParseLastBlockAndBlockType<InPipe>(bit_stream);

    last_block = last_block_tmp;
    bool is_uncompressed_block = (block_type == 0);
    bool is_static_huffman_block = (block_type == 1);
    bool is_dynamic_huffman_block = (block_type == 2);
    // END: parsing the first three bits
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // BEGIN: parsing the code length table (first table)
    ac_uint<9> numlitlencodes;
    ac_uint<6> numdistcodes;
    ac_uint<5> numcodelencodes;

    // These four tables are the first huffman table in optimized form, which
    // are used to decode the second Huffman table.
    // They are read from the ParseFirstTable helper function.
    [[intel::fpga_register]] ac_uint<8> codelencode_map_first_code[8];
    [[intel::fpga_register]] ac_uint<8> codelencode_map_last_code[8];
    [[intel::fpga_register]] ac_uint<5> codelencode_map_base_idx[8];
    [[intel::fpga_register]] ac_uint<5> codelencode_map[19];

    if (is_dynamic_huffman_block) {
      detail::ParseFirstTable<InPipe>(bit_stream, numlitlencodes, numdistcodes,
                                      numcodelencodes,
                                      codelencode_map_first_code,
                                      codelencode_map_last_code,
                                      codelencode_map_base_idx,
                                      codelencode_map);
    }

    // the number of literal and distance codes is known at compile time for
    // static Huffman blocks
    if (is_static_huffman_block) {
      numlitlencodes = kDeflateStaticNumLitLenCodes;
      numdistcodes = kDeflateStaticNumDistCodes;
    }
    // END: parsing the code length table (first table)
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // BEGIN: parsing the literal and distance tables (the second table)

    // These 8 tables are the second Huffman table in optimized form, which are
    // used to decode the actual payload data in optimized form. They are read
    // from the ParseSecondTable helper function.
    [[intel::fpga_register]] ac_uint<15> lit_map_first_code[15];
    [[intel::fpga_register]] ac_uint<15> lit_map_last_code[15];
    [[intel::fpga_register]] ac_uint<9> lit_map_base_idx[15];
    [[intel::fpga_register]] ac_uint<9> lit_map[286];
    [[intel::fpga_register]] ac_uint<15> dist_map_first_code[15];
    [[intel::fpga_register]] ac_uint<15> dist_map_last_code[15];
    [[intel::fpga_register]] ac_uint<5> dist_map_base_idx[15];
    [[intel::fpga_register]] ac_uint<5> dist_map[32];

    if (is_static_huffman_block || is_dynamic_huffman_block) {
      detail::ParseSecondTable<InPipe>(bit_stream,
                                       is_static_huffman_block,
                                       numlitlencodes,
                                       numdistcodes,
                                       numcodelencodes,
                                       codelencode_map_first_code,
                                       codelencode_map_last_code,
                                       codelencode_map_base_idx,
                                       codelencode_map,

                                       lit_map_first_code,
                                       lit_map_last_code,
                                       lit_map_base_idx,
                                       lit_map,
                                       dist_map_first_code,
                                       dist_map_last_code,
                                       dist_map_base_idx,
                                       dist_map);
    }
    // END: parsing the literal and distance tables (the second table)
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // BEGIN: decoding the bit stream (main computation loop)
    // indicates whether we are reading a distance (or literal) currently
    bool reading_distance = false;

    // true if the stop code (256) has been decoded and the block is done
    // track when a block is done:
    //    for compressed blocks (static and dynamic), when stop code is hit
    //    for uncompressed blocks, when all uncompressed bytes are read
    bool block_done = false;

    // true when output is ready. the output will be either a character or
    // a length distance pair (see HuffmanData struct)
    bool out_ready = false;

    // the output of this kernel goes to the LZ77 decoder
    GzipLZ77InputData<2> out_data;

    // for uncompressed blocks, the first 16 bits (after aligning to a byte)
    // is the length (in bytes) of the uncompressed data, followed by 16-bits
    // which is ~length (for error checking; we will ignore this).
    bool parsing_uncompressed_len = true;
    ac_uint<2> uncompressed_len_bytes_read = 0;
    unsigned short uncompressed_bytes_remaining;
    unsigned char first_four_bytes[4];

    // if this is an uncompressed block, we must realign to a byte boundary
    if (is_uncompressed_block) {
      bit_stream.AlignToByteBoundary();
    }

    ac_uint<9> lit_symbol;
    ac_uint<5> dist_symbol;

    // main processing loop
    // the II of this main loop can be controlled from the command line using
    // the -DHUFFMAN_II=<desired II>. By default, we let the
    // the compiler choose the Fmax/II to maximize throughput
#ifdef HUFFMAN_MAIN_LOOP_II
    [[intel::initiation_interval(HUFFMAN_II)]]  // NO-FORMAT: Attribute
#endif
    while (!block_done) {
      // read in new data if the ByteBitStream has space for it
      if (bit_stream.HasSpaceForByte()) {
        bool read_valid;
        auto pd = InPipe::read(read_valid);

        if (read_valid) {
          done_reading = pd.flag;
          bit_stream.NewByte(pd.data[0]);
        }
      }

      if (is_uncompressed_block) {
        // uncompressed block case
        if (bit_stream.Size() >= 8) {
          // grab a byte from the bit stream
          ac_uint<8> byte = bit_stream.ReadUInt<8>();

          if (parsing_uncompressed_len) {
            // first 16-bits are length, next 16-bits are ~length
            first_four_bytes[uncompressed_len_bytes_read] = byte.to_uint();
            if (uncompressed_len_bytes_read == 3) {
              // uncompressed_bytes_remaining = uncompressed_len
              // we will ignore uncompressed_len_n
              uncompressed_bytes_remaining =
                  (unsigned short)(first_four_bytes[1] << 8) |
                  (unsigned short)(first_four_bytes[0]);

              // done parsing uncompressed length
              parsing_uncompressed_len = false;
            }
            uncompressed_len_bytes_read += 1;
          } else {
            // for uncompressed blocks, simply read an 8-bit character from the
            // stream and write it to the output
            out_data.is_literal = true;
            out_data.literal[0] = byte;
            out_data.valid_count = 1;
            out_ready = true;
            uncompressed_bytes_remaining--;
            block_done = (uncompressed_bytes_remaining == 0);
          }

          bit_stream.Shift(ac_uint<4>(8));
        }
      } else if (bit_stream.Size() >= 30) {
        // dynamic of static huffman block, and we know we have at least 30 bits
        // read the next 30 bits (we know we have them)
        ac_uint<30> next_bits = bit_stream.ReadUInt<30>();

        // find all possible dynamic lengths
        [[intel::fpga_register]] ac_uint<5> lit_extra_bit_vals[15][5];
#pragma unroll
        for (int out_codelen = 1; out_codelen <= 15; out_codelen++) {
#pragma unroll
          for (int in_codelen = 1; in_codelen <= 5; in_codelen++) {
            ac_uint<5> codebits(0);
#pragma unroll
            for (int bit = 0; bit < in_codelen; bit++) {
              codebits[bit] = next_bits[out_codelen + bit] & 0x1;
            }
            lit_extra_bit_vals[out_codelen - 1][in_codelen - 1] = codebits;
          }
        }

        // find all possible dynamic distances
        [[intel::fpga_register]] ac_uint<15> dist_extra_bit_vals[15][15];
#pragma unroll
        for (int out_codelen = 1; out_codelen <= 15; out_codelen++) {
#pragma unroll
          for (int in_codelen = 1; in_codelen <= 15; in_codelen++) {
            ac_uint<15> codebits(0);
#pragma unroll
            for (int bit = 0; bit < in_codelen; bit++) {
              codebits[bit] = next_bits[out_codelen + bit] & 0x1;
            }
            dist_extra_bit_vals[out_codelen - 1][in_codelen - 1] = codebits;
          }
        }

        // find all possible code lengths and offsets
        // codes are 1 to 15 bits long, so we will do 15 of these computations
        // in parallel
        ac_uint<15> lit_codelen_valid_bitmap(0);
        ac_uint<15> dist_codelen_valid_bitmap(0);
        [[intel::fpga_register]] ac_uint<9> lit_codelen_offset[15];
        [[intel::fpga_register]] ac_uint<9> lit_codelen_base_idx[15];
        [[intel::fpga_register]] ac_uint<5> dist_codelen_offset[15];
        [[intel::fpga_register]] ac_uint<5> dist_codelen_base_idx[15];

#pragma unroll
        for (unsigned char codelen = 1; codelen <= 15; codelen++) {
          // Grab the 'codelen' bits we want and reverse them (little endian)
          ac_uint<15> codebits(0);
#pragma unroll
          for (unsigned char bit = 0; bit < codelen; bit++) {
            codebits[codelen - bit - 1] = next_bits[bit];
          }

          // for this code length, get the base index, first valid code, and
          // last valid code for both the literal and distance table
          auto lit_base_idx = lit_map_base_idx[codelen - 1];
          auto lit_first_code = lit_map_first_code[codelen - 1];
          auto lit_last_code = lit_map_last_code[codelen - 1];
          auto dist_base_idx = dist_map_base_idx[codelen - 1];
          auto dist_first_code = dist_map_first_code[codelen - 1];
          auto dist_last_code = dist_map_last_code[codelen - 1];

          // checking a literal match
          lit_codelen_valid_bitmap[codelen - 1] =
              ((codebits >= lit_first_code) && (codebits < lit_last_code)) ? 1
                                                                           : 0;
          lit_codelen_base_idx[codelen - 1] = lit_base_idx;
          lit_codelen_offset[codelen - 1] = codebits - lit_first_code;

          // checking a distance match
          dist_codelen_valid_bitmap[codelen - 1] =
              ((codebits >= dist_first_code) && (codebits < dist_last_code))
                  ? 1
                  : 0;
          dist_codelen_base_idx[codelen - 1] = dist_base_idx;
          dist_codelen_offset[codelen - 1] = codebits - dist_first_code;
        }

        // find the shortest matching length, which is the next decoded symbol
        ac_uint<4> lit_shortest_match_len_idx = CTZ(lit_codelen_valid_bitmap);
        ac_uint<4> lit_shortest_match_len =
            lit_shortest_match_len_idx + ac_uint<1>(1);
        ac_uint<4> dist_shortest_match_len_idx = CTZ(dist_codelen_valid_bitmap);
        ac_uint<4> dist_shortest_match_len =
            dist_shortest_match_len_idx + ac_uint<1>(1);

        // get the base index and offset based on the shortest match length
        auto lit_base_idx = lit_codelen_base_idx[lit_shortest_match_len_idx];
        auto lit_offset = lit_codelen_offset[lit_shortest_match_len_idx];
        auto dist_base_idx = dist_codelen_base_idx[dist_shortest_match_len_idx];
        auto dist_offset = dist_codelen_offset[dist_shortest_match_len_idx];
        ac_uint<9> lit_idx = lit_base_idx + lit_offset;
        ac_uint<9> dist_idx = dist_base_idx + dist_offset;

        // lookup the literal (literal or length) and distance using base_idx
        // and offset
        lit_symbol = lit_map[lit_idx];
        dist_symbol = dist_map[dist_idx];

        // we will either shift by shortest_match_len or by
        // shortest_match_len + num_extra_bits based on whether we read a
        // length, distance and/or extra bits.
        // maximum value for shift_amount = 15 + 15 = 30, ceil(log2(30)) = 5
        ac_uint<5> shift_amount;

        if (!reading_distance) {
          shift_amount = lit_shortest_match_len;
          // currently parsing a symbol or length (same table)
          if (lit_symbol == 256) {
            // stop code hit, done this block
            block_done = true;
          } else if (lit_symbol < 256) {
            // decoded a literal (i.e. not a length)
            out_data.is_literal = true;
            out_data.literal[0] = lit_symbol;
            out_data.valid_count = 1;
            out_ready = true;
          } else if (lit_symbol <= 264) {
            // decoded a length that doesn't require us to read more bits
            out_data.is_literal = false;
            out_data.length = lit_symbol - ac_uint<9>(254);
            reading_distance = true;
          } else if (lit_symbol <= 284) {
            // decoded a length that requires us to read more bits
            ac_uint<5> lit_symbol_small = lit_symbol.template slc<5>(0);
            // (lit_symbol - 261) / 4
            ac_uint<3> num_extra_bits = (lit_symbol_small - ac_uint<5>(5)) >> 2;
            auto extra_bits_val = lit_extra_bit_vals[lit_shortest_match_len_idx]
                                                    [num_extra_bits - 1];
            // ((((lit_symbol - 265) % 4) + 4) << num_extra_bits) +
            // ac_uint<2>(3) + extra_bits_val
            out_data.is_literal = false;
            out_data.length =
                ((((lit_symbol_small - ac_uint<5>(9)) & 0x3) + ac_uint<3>(4))
                 << num_extra_bits) +
                ac_uint<2>(3) + extra_bits_val;
            shift_amount = lit_shortest_match_len + num_extra_bits;
            reading_distance = true;
          } else if (lit_symbol == 285) {
            // decoded a length that doesn't require us to read more bits
            out_data.is_literal = false;
            out_data.length = 258;
            reading_distance = true;
          }  // else error, ignored
        } else {
          shift_amount = dist_shortest_match_len;
          // currently decoding a distance symbol
          if (dist_symbol <= 3) {
            // decoded a distance that doesn't require us to read more bits
            out_data.distance = dist_symbol + ac_uint<1>(1);
          } else {
            // decoded a distance that requires us to read more bits
            ac_uint<4> num_extra_bits = (dist_symbol >> 1) - ac_uint<1>(1);
            auto extra_bits_val =
                dist_extra_bit_vals[dist_shortest_match_len_idx]
                                   [num_extra_bits - 1];
            out_data.distance =
                (((dist_symbol & 0x1) + ac_uint<2>(2)) << num_extra_bits) +
                ac_uint<1>(1) + extra_bits_val;
            shift_amount = dist_shortest_match_len + num_extra_bits;
          }
          out_ready = true;
          reading_distance = false;
        }

        // shift based on how many bits we read
        bit_stream.Shift(shift_amount);
      }

      // output data to downstream kernel when ready
      if (out_ready) {
        OutPipe::write(OutPipeBundleT(out_data));
        out_ready = false;
      }
    }  // while (!block_done)
  }    // while (!last_block)
  // END: decoding the bit stream (main computation loop)
  ////////////////////////////////////////////////////////////////////////////

  // notify the downstream kernel that we are done
  OutPipe::write(OutPipeBundleT(true));

  // read out the remaining data from the pipe
  // NOTE: don't really care about performance here since it reads out 8 bytes
  // for CRC-32 and uncompressed size and maybe one more byte for padding.
  while (!done_reading) {
    bool read_valid;
    auto pd = InPipe::read(read_valid);
    done_reading = pd.flag && read_valid;
  }
}

//
// Creates a kernel from the Huffman decoder function
//
template <typename Id, typename InPipe, typename OutPipe>
sycl::event SubmitHuffmanDecoder(sycl::queue& q) {
  return q.single_task<Id>([=] {
    HuffmanDecoder<InPipe, OutPipe>();
  });
}

// helper functions for parsing the block headers
namespace detail {

//
// Parses the first 3 bits of the DEFLATE block returns their meaning:
//    The first bit indicates whether this is the last block in a stream
//    The following two bits indicate the block type:
//        0: uncompressed
//        1: static huffman
//        2: dynamic huffman
//        3: reserved
//
template <typename InPipe>
std::pair<bool, ac_uint<2>> ParseLastBlockAndBlockType(BitStreamT& bit_stream) {
  // read in the first byte and add it to the byte bit stream
  auto first_pipe_data = InPipe::read();
  bit_stream.NewByte(first_pipe_data.data[0]);

  // read the first three bits
  ac_uint<3> first_three_bits = bit_stream.ReadUInt<3>();
  bit_stream.Shift(3);

  // first bit indicates whether this is the last block
  bool last_block = (first_three_bits.slc<1>(0) == 1);

  // next 2 bits indicate the block type
  ac_uint<2> block_type = first_three_bits.slc<2>(1);

  return std::make_pair(last_block, block_type);
}


//
// Parses the first Huffman table and creates the optimized Huffman table
// structure
//
template <typename InPipe>
void ParseFirstTable(BitStreamT& bit_stream,

                     // outputs
                     ac_uint<9>& numlitlencodes,
                     ac_uint<6>& numdistcodes,
                     ac_uint<5>& numcodelencodes,
                     ac_uint<8> codelencode_map_first_code[8],
                     ac_uint<8> codelencode_map_last_code[8],
                     ac_uint<5> codelencode_map_base_idx[8],
                     ac_uint<5> codelencode_map[19]) {
  ac_uint<2> first_table_state = 0;
  unsigned short codelencodelen_count = 0;

  // shuffle vector for the first table
  constexpr unsigned short codelencodelen_idxs[] = {
      16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

  [[intel::fpga_register]] ac_uint<3> codelencodelen[19];
#pragma unroll
  for (int i = 0; i < 19; i++) {
    codelencodelen[i] = 0;
  }

  bool parsing = true;

  // NOTE: this loop is not the main processing loop and therefore is
  // not critical (low trip count). However, the compiler doesn't know that
  // and tries to optimize for throughput (~Fmax/II). We don't want
  // this loop to be our Fmax bottleneck, so increase the II.
  [[intel::initiation_interval(3)]]  // NO-FORMAT: Attribute
  while (parsing) {
    // grab a byte if we have space for it
    if (bit_stream.HasSpaceForByte()) {
      bool read_valid;
      auto pd = InPipe::read(read_valid);

      if (read_valid) {
        unsigned char c = pd.data[0];
        bit_stream.NewByte(c);
      }
    }

    // make sure we have enough bits (in the maximum case)
    if (bit_stream.Size() >= 5) {
      if (first_table_state == 0) {
        // read the number of literal length codes
        numlitlencodes = bit_stream.ReadUInt(5) + ac_uint<9>(257);
        bit_stream.Shift(5);
        first_table_state = 1;
      } else if (first_table_state == 1) {
        // read the number of distance length codes
        numdistcodes = bit_stream.ReadUInt(5) + ac_uint<1>(1);
        bit_stream.Shift(5);
        first_table_state = 2;
      } else if (first_table_state == 2) {
        // read the number of code length codes (for encoding code lengths)
        numcodelencodes = bit_stream.ReadUInt(4) + ac_uint<3>(4);
        bit_stream.Shift(4);
        first_table_state = 3;
      } else if (codelencodelen_count < numcodelencodes) {
        // read the code lengths themselves
        auto tmp = bit_stream.ReadUInt(3);
        bit_stream.Shift(3);
        codelencodelen[codelencodelen_idxs[codelencodelen_count]] = tmp;
        codelencodelen_count++;
        parsing = (codelencodelen_count != numcodelencodes);
      }
    }
  }

  ac_uint<8> codelencode_map_next_code = 0;
  ac_uint<5> codelencode_map_counter = 0;
  for (unsigned char codelen = 1; codelen <= 8; codelen++) {
    codelencode_map_next_code <<= 1;
    codelencode_map_first_code[codelen - 1] = codelencode_map_next_code;
    codelencode_map_base_idx[codelen - 1] = codelencode_map_counter;
    for (unsigned short symbol = 0; symbol < 19; symbol++) {
      auto inner_codelen = codelencodelen[symbol];
      if (inner_codelen == codelen) {
        codelencode_map[codelencode_map_counter] = symbol;
        codelencode_map_counter++;
        codelencode_map_next_code++;
      }
    }
    codelencode_map_last_code[codelen - 1] = codelencode_map_next_code;
  }
}

//
// Parses the second Huffman table and creates the optimized Huffman table
// structure
//
template <typename InPipe>
void ParseSecondTable(BitStreamT& bit_stream,
                      bool is_static_huffman_block,
                      ac_uint<9> numlitlencodes,
                      ac_uint<6> numdistcodes,
                      ac_uint<5> numcodelencodes,
                      ac_uint<8> codelencode_map_first_code[8],
                      ac_uint<8> codelencode_map_last_code[8],
                      ac_uint<5> codelencode_map_base_idx[8],
                      ac_uint<5> codelencode_map[19],

                      // outputs
                      ac_uint<15> lit_map_first_code[15],
                      ac_uint<15> lit_map_last_code[15],
                      ac_uint<9> lit_map_base_idx[15],
                      ac_uint<9> lit_map[286],
                      ac_uint<15> dist_map_first_code[15],
                      ac_uint<15> dist_map_last_code[15],
                      ac_uint<5> dist_map_base_idx[15],
                      ac_uint<5> dist_map[32]) {
  // length of codelens is MAX(numlitlencodes + numdistcodes)
  // = MAX((2^5 + 257) + (2^5 + 1)) = 322
  // maximum code length = 15, so requires 4 bits to store
  ac_uint<4> codelens[322];
  ac_uint<9> total_codes_second_table = numlitlencodes + numdistcodes;
  decltype(total_codes_second_table) codelens_idx = 0;
  bool decoding_next_symbol = true;
  ac_uint<8> runlen;  // MAX = (2^7 + 11)
  int onecount = 0, otherpositivecount = 0;
  ac_uint<4> extend_symbol;

  // static codelens ROM (for static huffman encoding)
  constexpr auto static_codelens = [] {
    std::array<unsigned short, 320> a{};
    // literal codes
    for (int i = 0; i < kDeflateStaticNumLitLenCodes; i++) {
      if (i < 144) {
        a[i] = 8;
      } else if (i < 144 + 112) {
        a[i] = 9;
      } else if (i < 144 + 112 + 24) {
        a[i] = 7;
      } else {
        a[i] = 8;
      }
    }
    // distance codes
    for (int i = 0; i < kDeflateStaticNumDistCodes; i++) {
      a[kDeflateStaticNumLitLenCodes + i] = 5;
    }
    return a;
  }();

  if (is_static_huffman_block) {
    // for a static huffman block, initialize codelens with static codelens ROM
    for (int i = 0; i < kDeflateStaticTotalCodes; i++) {
      codelens[i] = static_codelens[i];
    }
  } else {  // is_dynamic_huffman_block
    // NOTE: this loop is not the main processing loop and therefore is
    // not critical (low trip count). However, the compiler doesn't know that
    // and tries to optimize for throughput (~Fmax/II). We don't want
    // this loop to be our Fmax bottleneck, so increase the II.
    [[intel::initiation_interval(3)]]  // NO-FORMAT: Attribute
    while (codelens_idx < total_codes_second_table) {
      // read in another byte if we have space for it
      if (bit_stream.HasSpaceForByte()) {
        bool read_valid;
        auto pd = InPipe::read(read_valid);

        if (read_valid) {
          bit_stream.NewByte(pd.data[0]);
        }
      }

      if (decoding_next_symbol) {
        // decoding the next code symbol, so make sure we have enough bits to
        // do so 15 bits is the maximum bits to read both a symbol and the
        // extra run length bits (max 8 bits for the symbol, max 7 bits for
        // extra run length)
        if (bit_stream.Size() >= 15) {
          // read 15 bits
          ac_uint<15> next_bits = bit_stream.ReadUInt<15>();

          // find all possible dynamic run lengths
          // the symbol could be from 1 to 8 bits long and the number of extra
          // bits to read for the run length could be either 2, 3, or 7 bits
          // (the 3 possibilities in 'runlen_bits').
          [[intel::fpga_register]] ac_uint<7> extra_bit_vals[8][3];
          constexpr int runlen_bits[] = {2, 3, 7};
#pragma unroll
          for (int out_codelen = 1; out_codelen <= 8; out_codelen++) {
#pragma unroll
            for (int j = 0; j < 3; j++) {
              ac_uint<7> codebits(0);
#pragma unroll
              for (int bit = 0; bit < runlen_bits[j]; bit++) {
                codebits[bit] = next_bits[out_codelen + bit] & 0x1;
              }
              extra_bit_vals[out_codelen - 1][j] = codebits;
            }
          }

          // decode all possible code symbols from 1 to 8 bits
          ac_uint<8> codelencode_valid_bitmap(0);
          [[intel::fpga_register]] ac_uint<5> codelencode_offset[8];
          [[intel::fpga_register]] ac_uint<5> codelencode_base_idx[8];

#pragma unroll
          for (int codelen = 1; codelen <= 8; codelen++) {
            ac_uint<8> codebits(0);
#pragma unroll
            for (int bit = 0; bit < codelen; bit++) {
              codebits[codelen - bit - 1] = next_bits[bit] & 0x1;
            }

            auto base_idx = codelencode_map_base_idx[codelen - 1];
            auto first_code = codelencode_map_first_code[codelen - 1];
            auto last_code = codelencode_map_last_code[codelen - 1];

            codelencode_base_idx[codelen - 1] = base_idx;
            codelencode_valid_bitmap[codelen - 1] =
                ((codebits >= first_code) && (codebits < last_code)) ? 1 : 0;

            codelencode_offset[codelen - 1] = codebits - first_code;
          }

          // find the shortest matching code symbol
          ac_uint<3> shortest_match_len_idx = CTZ(codelencode_valid_bitmap);
          ac_uint<3> shortest_match_len =
              shortest_match_len_idx + ac_uint<1>(1);
          ac_uint<5> base_idx = codelencode_base_idx[shortest_match_len_idx];
          ac_uint<5> offset = codelencode_offset[shortest_match_len_idx];

          // get the decoded symbol
          auto symbol = codelencode_map[base_idx + offset];

          // max shift amount will be 15 (8 bits for symbol, 7 for run length)
          ac_uint<4> shift_amount;

          // do logic based on symbol value
          if (symbol <= 15) {
            // ADD SYMBOL
            codelens[codelens_idx++] = symbol;
            decoding_next_symbol = true;
            if (codelens_idx >= numlitlencodes) {
              if (symbol == 1) {
                onecount++;
              } else if (symbol > 0) {
                otherpositivecount++;
              }
            }
            shift_amount = shortest_match_len;
          } else if (symbol == 16) {
            // READ 2-BIT RUN LENGTH, ADD 3, AND EXTEND LAST ELEMENT
            runlen = extra_bit_vals[shortest_match_len_idx][0] + ac_uint<2>(3);
            decoding_next_symbol = false;
            extend_symbol = codelens[codelens_idx - 1];
            shift_amount = shortest_match_len + ac_uint<2>(2);
          } else if (symbol == 17) {
            // READ 3-BIT RUN LENGTH, ADD 3, AND EXTEND WITH 0's
            runlen = extra_bit_vals[shortest_match_len_idx][1] + ac_uint<2>(3);
            decoding_next_symbol = false;
            extend_symbol = 0;
            shift_amount = shortest_match_len + ac_uint<2>(3);
          } else if (symbol == 18) {
            // READ 7-BIT RUN LENGTH, ADD 11, AND EXTEND WITH 0's
            runlen = extra_bit_vals[shortest_match_len_idx][2] + ac_uint<4>(11);
            decoding_next_symbol = false;
            extend_symbol = 0;
            shift_amount = shortest_match_len + ac_uint<3>(7);
          }

          // shift the bit stream
          bit_stream.Shift(shift_amount);
        }
      } else {
        // extending codelens
        codelens[codelens_idx++] = extend_symbol;
        if (codelens_idx >= numlitlencodes) {
          if (extend_symbol == 1) {
            onecount++;
          } else if (extend_symbol > 0) {
            otherpositivecount++;
          }
        }

        // decrement the run length
        runlen--;

        // start reading decoding symbols again when runlen == 0
        decoding_next_symbol = (runlen == 0);
      }
    }

    // handle the case where only one distance code is defined add a dummy
    // invalid code to make the Huffman tree complete
    if (onecount == 1 && otherpositivecount == 0) {
      int extend_amount = 32 - numdistcodes;
      for (int i = 0; i < extend_amount; i++) {
        codelens[numlitlencodes + numdistcodes + i] = 0;
      }
      codelens[numlitlencodes + 31] = 1;
      numdistcodes += extend_amount;
    }
  }

  // the first table is decoded, so now it is time to decode the second
  // table, which is actually two tables:
  //  literal table (symbols and lengths for the {length, distance} pair)
  //  distance table (the distances for the {length, distance} pair)
  //
  // NOTE ON OPTIMIZATION: The DEFLATE algorithm defines the Huffman table
  // by just storing the number of bits for each symbol. This method to
  // convert from a list of code lengths to the Huffman table is such that
  // codes of the same length (in bits) are sequential.
  // Since the range of code length bits is [1, 15] bits, we can
  // efficiently store the huffman table as follows:
  // - Store the first and last code for each code length.
  // - Store a map from the index to symbol in a map (lit_map and dist_map)
  // - Store the base index for each code length into the map
  // Then, given N bits, with a value of V, you can check for a match like so:
  //    bool match = (V >= first_code[N]) && (V < last_code[N]);
  // And, if we get a match for this code, we can get the symbol like so:
  //    int offset = V - first_code[N];
  //    symbol = map[base_idx[N] + offset];
  //
  // This structure is the same for both the 'lit_map' and 'dist_map' below,
  // the only difference being there are 286 possibilities for literals
  // and 32 for distances. When decoding, the presence of a 'length' literal
  // implies the next thing we decode is a distance.

  // the maximum of numdistcodes and numlitlencodes
  ac_uint<9> max_codes;
  if (numdistcodes < numlitlencodes)
    max_codes = numlitlencodes;
  else
    max_codes = numdistcodes;

  ac_uint<15> lit_map_next_code = 0;
  ac_uint<9> lit_map_counter = 0;
  ac_uint<15> dist_map_next_code = 0;
  ac_uint<5> dist_map_counter = 0;
  for (unsigned char codelen = 1; codelen <= 15; codelen++) {
    lit_map_next_code <<= 1;
    lit_map_first_code[codelen - 1] = lit_map_next_code;
    lit_map_base_idx[codelen - 1] = lit_map_counter;

    dist_map_next_code <<= 1;
    dist_map_first_code[codelen - 1] = dist_map_next_code;
    dist_map_base_idx[codelen - 1] = dist_map_counter;

    // this loop has been manually fused for the literal and distance codes
    // we will iterate max(numdistcodes, numlitlencodes) times and predicate
    // the decoding based on numdistcodes and numlitlencodes
    for (unsigned short symbol = 0; symbol < max_codes; symbol++) {
      // literal
      if (symbol < numlitlencodes) {
        auto inner_codelen = codelens[symbol];
        if (inner_codelen == codelen) {
          lit_map[lit_map_counter] = symbol;
          lit_map_counter++;
          lit_map_next_code++;
        }
      }

      // distance
      if (symbol < numdistcodes) {
        auto inner_codelen = codelens[numlitlencodes + symbol];
        if (inner_codelen == codelen) {
          dist_map[dist_map_counter] = symbol;
          dist_map_counter++;
          dist_map_next_code++;
        }
      }
    }

    lit_map_last_code[codelen - 1] = lit_map_next_code;
    dist_map_last_code[codelen - 1] = dist_map_next_code;
  }
}

}  // namespace detail

#endif /* __HUFFMAN_DECODER_HPP__ */