#ifndef __DECOMPRESSOR_HPP__
#define __DECOMPRESSOR_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "ByteBitStream.hpp"
#include "HeaderData.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

// we only use unsigned ac_ints, so use this alias to avoid having to type
// 'false' all the time
template<int bits>
using ac_uint = ac_int<bits, false>;

#ifdef BIT_BUFFER_BITS
constexpr int kBitBufferBits = BIT_BUFFER_BITS;
#else
constexpr int kBitBufferBits = 48;
#endif
constexpr int kBitBufferMaxReadBits = 5;
constexpr int kBitBufferMaxShiftBits = 30;
using BitStreamT =
  ByteBitStream<kBitBufferBits, kBitBufferMaxReadBits, kBitBufferMaxShiftBits>;

template<int bits> 
void PrintUACInt(ac_uint<bits>& x) {
  for (int i = bits-1; i >= 0; i--) {
    PRINTF("%d", x[i] & 0x1);
  }
}

//
// TODO
//
template<typename T>
struct FlagBundle {
  FlagBundle() : data(T()), flag(false) {}
  FlagBundle(T d_in) : data(d_in), flag(false) {}
  FlagBundle(T d_in, bool f_in) : data(d_in), flag(f_in) {}
  FlagBundle(bool f_in) : data(T()), flag(f_in) {}

  T data;
  bool flag;
};

//
// TODO
//
struct HuffmanData {
  HuffmanData() : len_or_sym(0), dist_or_flag(0) {}
  HuffmanData(short symbol) : len_or_sym(symbol), dist_or_flag(-1) {}
  HuffmanData(short len_in, short dist_in) :
    len_or_sym(len_in), dist_or_flag(dist_in) {}
  short len_or_sym;
  short dist_or_flag; 
};

//
// TODO
//
template<int n>
struct LiteralPack {
  static constexpr int count_bits = fpga_tools::Log2(n) + 1;
  unsigned char byte[n];  // TODO: rename?
  ac_uint<count_bits> valid_count;
};

//
// TODO
//
template<int bits>
auto ctz(const ac_uint<bits>& in) {
    constexpr int out_bits = fpga_tools::Log2(bits) + 1;
    //ac_uint<out_bits> ret(bits);
    ac_uint<out_bits> ret;
    #pragma unroll
    for (int i = bits - 1; i >= 0; i--) {
      if (in[i]) {
        ret = i;
      }
    }
    return ret;
}

class HeaderKernelID;
class HuffmanDecoderKernelID;
class LZ77DecoderKernelID;
class LiteralStackerKernelID;

class HeaderToHuffmanPipeID;
class HuffmanToLZ77PipeID;
class LZ77ToLiteralStackerPipeID;

template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
event SubmitLiteralStackerKernel(queue& q) {
  return q.single_task<LiteralStackerKernelID>([=] {
    using OutPipeBundleT = FlagBundle<LiteralPack<literals_per_cycle>>;
    constexpr int cache_idx_bits = fpga_tools::Log2(literals_per_cycle*2) + 1;

    bool done;

    // NOTE: cache_idx is fmax bottleneck of this kernel
    // could shannonize to improve, but not worth it yet
    ac_uint<cache_idx_bits> cache_idx = 0;
    [[intel::fpga_register]] unsigned char cache_buf[literals_per_cycle * 2];

    do {
      bool data_valid;
      auto pipe_data = InPipe::read(data_valid);
      done = pipe_data.flag && data_valid;

      if (data_valid && !done) {
        #pragma unroll
        for (int i = 0; i < literals_per_cycle; i++) {
          if (i < pipe_data.data.valid_count) {
            cache_buf[cache_idx + i] = pipe_data.data.byte[i];
          }
        }
        cache_idx += pipe_data.data.valid_count;
      }

      if (cache_idx >= literals_per_cycle || done) {
        // create the output pack of characters from the current cache
        LiteralPack<literals_per_cycle> out_pack;
        #pragma unroll
        for (int i = 0; i < literals_per_cycle; i++) {
          // copy the character
          out_pack.byte[i] = cache_buf[i];

          // shift the extra characters to the front of the cache
          cache_buf[i] = cache_buf[i + literals_per_cycle];
        }

        // mark output with number of valid bytes
        if (cache_idx <= literals_per_cycle) {
          out_pack.valid_count = cache_idx;
        } else {
          out_pack.valid_count = literals_per_cycle;
        }

        // decrement cache_idx by number of elements we read
        // it is safe to always subtract literals_per_cycle since that will only
        // happen on the last iteration of the outer while loop (when 'done'
        // is true) 
        cache_idx -= literals_per_cycle;

        // write output
        OutPipe::write(OutPipeBundleT(out_pack));
      }
    } while (!done);
    
    // notify downstream kernel that we are done
    OutPipe::write(OutPipeBundleT(true));
  });
}

/*
template<typename InPipe, typename OutPipe>
event SubmitLZ77DecoderKernel(queue& q) {
  return q.single_task<LZ77DecoderKernelID>([=] {
    // the maximum history is defined by the DEFLATE algorithm
    // do not change this:
    //    making it smaller will make the design not functionally correct
    //    making it larger will waste space since the compressor follows
    //    this rule too
    constexpr unsigned kMaxHistory = 32768;
    constexpr unsigned kMaxHistoryMask = kMaxHistory - 1;

    // use a ring buffer for the history
    unsigned short history_idx = 0;
    unsigned char history[kMaxHistory];

    // history buffer shift register cache
    constexpr int kCacheDepth = 4;
    [[intel::fpga_register]] unsigned char history_cache_val[kCacheDepth + 1];
    [[intel::fpga_register]] unsigned short history_cache_idx[kCacheDepth + 1];

    bool done;
    bool reading_history = false;
    bool reading_history_next;
    unsigned short history_read_idx;
    unsigned short history_count;

    [[intel::ivdep(kCacheDepth)]]
    do {
      bool data_valid = true;
      unsigned char c;

      // if we aren't currently reading from the history, read from input pipe
      if (!reading_history) {
        // read from pipe
        auto pipe_data = InPipe::read(data_valid);

        // check if we are done
        done = pipe_data.flag && data_valid;

        // grab the symbol or the length and distance pair
        auto len_or_sym = pipe_data.data.len_or_sym;
        auto dist = pipe_data.data.dist_or_flag;
        c = len_or_sym & 0xFF;

        // if we get a length distance pair we will read 'len_or_sym' bytes
        // starting 'dist' 
        history_count = len_or_sym;
        reading_history = (dist != -1) && data_valid;
        reading_history_next = history_count != 1;
        history_read_idx = (history_idx - dist) & kMaxHistoryMask;
      }

      if (reading_history) {
        // grab from the history buffer
        c = history[history_read_idx];

        // also check the cache to see if it is there
        #pragma unroll
        for (int i = 0; i < kCacheDepth + 1; i++) {
          if (history_cache_idx[i] == history_read_idx) {
            c = history_cache_val[i];
          }
        }

        // update the history read index
        history_read_idx = (history_read_idx + 1) & kMaxHistoryMask;

        // update whether we are still reading the history
        reading_history = reading_history_next;
        reading_history_next = history_count != 2;
        history_count--;
      }
      

      if (!done && data_valid) {
        // write the new value to both the history buffer and the cache
        history[history_idx] = c;
        history_cache_val[kCacheDepth] = c;
        history_cache_idx[kCacheDepth] = history_idx;

        // Cache is just a shift register, so shift the shift reg. Pushing
        // into the back of the shift reg is done above.
        #pragma unroll
        for (int i = 0; i < kCacheDepth; i++) {
          history_cache_val[i] = history_cache_val[i + 1];
          history_cache_idx[i] = history_cache_idx[i + 1];
        }

        // update the history index
        history_idx = (history_idx + 1) & kMaxHistoryMask;

        // write the output to the pipe
        OutPipe::write(FlagBundle<unsigned char>(c));
      }
    } while (!done);
    
    OutPipe::write(FlagBundle<unsigned char>(0, true));
  });
}
*/

template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
event SubmitLZ77DecoderKernel(queue& q) {
  return q.single_task<LZ77DecoderKernelID>([=] {
    static_assert(literals_per_cycle > 0);
    static_assert(fpga_tools::IsPow2(literals_per_cycle));
    using OutPipeBundleT = FlagBundle<LiteralPack<literals_per_cycle>>;

    bool done;
    bool reading_history = false;
    bool reading_history_next;
    short history_counter;

    constexpr unsigned kMaxHistory = 32768;
    constexpr unsigned history_buffer_count = kMaxHistory / literals_per_cycle;
    constexpr unsigned history_buffer_mask = history_buffer_count - 1;
    constexpr unsigned history_buffer_idx_bits = fpga_tools::Log2(literals_per_cycle);
    constexpr unsigned history_buffer_idx_mask = literals_per_cycle - 1;

    unsigned history_buffer_buffer_idx = 0;
    [[intel::fpga_register]] unsigned history_buffer_idx[literals_per_cycle];
    unsigned char history_buffer[literals_per_cycle][history_buffer_count];

    unsigned read_history_buffer_buffer_idx = 0;
    [[intel::fpga_register]] unsigned read_history_buffer_idx[literals_per_cycle];
    [[intel::fpga_register]] int read_history_buffer_feedback_idx[literals_per_cycle];

    #pragma unroll
    for (int i = 0; i < literals_per_cycle; i++) { history_buffer_idx[i] = 0; }

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

        out_data.byte[0] = len_or_sym & 0xFF;
        out_data.valid_count = 1;

        // if we get a length distance pair we will read 'len_or_sym' bytes
        // starting 'dist'
        history_counter = len_or_sym;
        reading_history = (dist > 0) && data_valid;
        reading_history_next = history_counter > literals_per_cycle;

        // set history_read_idx for each buffer
        unsigned udist = dist;
        //short low_dist_bits = dist & history_buffer_idx_mask;
        read_history_buffer_buffer_idx = (history_buffer_buffer_idx - udist) & history_buffer_idx_mask;
        #pragma unroll
        for (int i = 0; i < literals_per_cycle; i++) {
          unsigned buf_idx = (read_history_buffer_buffer_idx + i) & history_buffer_idx_mask;
          if (dist - i > 0) {
            unsigned starting_read_idx_for_this_buf = (history_buffer_idx[buf_idx] - ((dist - i) / literals_per_cycle)) - 1;
            if (buf_idx == history_buffer_buffer_idx) {
              starting_read_idx_for_this_buf += 1;
            }
            read_history_buffer_idx[buf_idx] = starting_read_idx_for_this_buf & history_buffer_mask;
            read_history_buffer_feedback_idx[buf_idx] = -1;
          } else {
            read_history_buffer_idx[buf_idx] = 0;
            read_history_buffer_feedback_idx[buf_idx] = (i - dist);
          }
        }
      }

      if (reading_history) {
        // grab from the history buffer
        // TODO: read at {0, 1, 2, 3} and shuffle
        #pragma unroll
        for (int i = 0; i < literals_per_cycle; i++) {
          unsigned buf_idx = (read_history_buffer_buffer_idx + i) & history_buffer_idx_mask;
          unsigned idx_in_buf = read_history_buffer_idx[buf_idx];
          int idx_in_prev = read_history_buffer_feedback_idx[buf_idx];
          if (idx_in_prev < 0) {
            out_data.byte[i] = history_buffer[buf_idx][idx_in_buf];
          } else {
            out_data.byte[i] = out_data.byte[idx_in_prev];
          }
        }
        out_data.valid_count =
          (history_counter < literals_per_cycle) ? history_counter : literals_per_cycle;

        // update the history read index
        #pragma unroll
        for (int i = 0; i < literals_per_cycle; i++) {
          read_history_buffer_idx[i] = (read_history_buffer_idx[i] + 1) & history_buffer_mask;
        }

        // update whether we are still reading the history
        reading_history = reading_history_next;
        reading_history_next = history_counter > literals_per_cycle * 2;
        history_counter -= literals_per_cycle;
      }
      
      if (!done && data_valid) {
        // write to the history buffers what we are sending out
        #pragma unroll
        for (int i = 0; i < literals_per_cycle; i++) {
          if (i < out_data.valid_count) {
            unsigned buf_idx = (history_buffer_buffer_idx + i) & history_buffer_idx_mask;
            unsigned idx_in_buf = history_buffer_idx[buf_idx];
            history_buffer[buf_idx][idx_in_buf] = out_data.byte[i];
            history_buffer_idx[buf_idx] = (history_buffer_idx[buf_idx] + 1) & history_buffer_mask;
          }
        }

        // update the most recent buffer
        history_buffer_buffer_idx = (history_buffer_buffer_idx + out_data.valid_count) & history_buffer_idx_mask;

        // write the output to the pipe
        OutPipe::write(OutPipeBundleT(out_data));
      }
    } while (!done);
    
    OutPipe::write(OutPipeBundleT(true));
  });
}

template<typename InPipe, typename OutPipe>
event SubmitHuffmanDecoderKernel(queue& q) {
  return q.single_task<HuffmanDecoderKernelID>([=] {
    using OutPipeBundleT = FlagBundle<HuffmanData>;

    BitStreamT bbs;
    bool last_block;
    bool done_reading = false;

    [[intel::disable_loop_pipelining]]
    do {
      ac_uint<3> first_table_state = 0;
      ac_uint<1> last_block_num;
      ac_uint<2> type;
      ac_uint<9> numlitlencodes;
      ac_uint<6> numdistcodes;
      ac_uint<5> numcodelencodes;
      bool parsing_first_table = true;
      unsigned short codelencodelen_count = 0;

      constexpr unsigned short codelencodelen_idxs[] =
        {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

      [[intel::fpga_register]] ac_uint<3> codelencodelen[19];
      #pragma unroll
      for (int i = 0; i < 19; i++) { codelencodelen[i] = 0; }

      // NOTE: this loop is not the main processing loop and therefore is
      // not critical (low trip count). However, the compiler doesn't know that
      // and tries to optimize for throughput (~Fmax/II). However, we don't want
      // this loop to be our Fmax bottleneck, so increase the II.
      [[intel::initiation_interval(4)]]
      do {
        if (bbs.HasSpaceForByte()) {
          bool read_valid;
          auto pd = InPipe::read(read_valid);

          if (read_valid) {
            unsigned char c = pd.data;
            bbs.NewByte(c);
          }
        }

        if (bbs.Size() >= 5) {
          if (first_table_state == 0) {
            last_block_num = bbs.ReadUInt(1);
            //PRINTF("last_block_num: %u\n", last_block_num);
            bbs.Shift(1);
            last_block = (last_block_num & 0x1);
            first_table_state = 1;
          } else if (first_table_state == 1) {
            type = bbs.ReadUInt(2);
            //PRINTF("type: %u\n", type);
            bbs.Shift(2);
            first_table_state = 2;
          } else if (first_table_state == 2) {
            numlitlencodes = bbs.ReadUInt(5) + ac_uint<9>(257);
            //PRINTF("numlitlencodes: %u\n", numlitlencodes);
            bbs.Shift(5);
            first_table_state = 3;
          } else if (first_table_state == 3) {
            numdistcodes = bbs.ReadUInt(5) + ac_uint<1>(1);
            //PRINTF("numdistcodes: %u\n", numdistcodes);
            bbs.Shift(5);
            first_table_state = 4;
          } else if (first_table_state == 4) {
            numcodelencodes = bbs.ReadUInt(4) + ac_uint<3>(4);
            //PRINTF("numcodelencodes: %u\n", numcodelencodes);
            bbs.Shift(4);
            first_table_state = 5;
          } else if (codelencodelen_count < numcodelencodes) {
            auto tmp = bbs.ReadUInt(3);
            bbs.Shift(3);
            codelencodelen[codelencodelen_idxs[codelencodelen_count]] = tmp;
            //PRINTF("codelencodelen[%u] = %u\n", codelencodelen_idxs[codelencodelen_count], tmp);
            codelencodelen_count++;
            parsing_first_table = (codelencodelen_count != numcodelencodes);
          }
        }
      } while (parsing_first_table);

      [[intel::fpga_register]] ac_uint<8> codelencode_map_first_code[8];
      [[intel::fpga_register]] ac_uint<8> codelencode_map_last_code[8];
      [[intel::fpga_register]] ac_uint<5> codelencode_map_base_idx[8];
      [[intel::fpga_register]] ac_uint<5> codelencode_map[19];

      // std::remove_extent_t<decltype(codelencode_map_first_code)>
      // std::remove_extent_t<decltype(codelencode_map_base_idx)>
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

      // length of codelens is MAX(numlitlencodes + numdistcodes)
      // = MAX((2^5 + 257) + (2^5 + 1)) = 322
      ac_uint<15> codelens[322];
      ac_uint<9> total_codes_second_table = numlitlencodes + numdistcodes;
      decltype(total_codes_second_table) codelens_idx = 0;
      bool decoding_next_symbol = true;
      ac_uint<8> runlen; // MAX = (2^7 + 11)
      int onecount = 0, otherpositivecount = 0;
      ac_uint<15> extend_symbol;

      // NOTE: this loop is not the main processing loop and therefore is
      // not critical (low trip count). However, the compiler doesn't know that
      // and tries to optimize for throughput (~Fmax/II). However, we don't want
      // this loop to be our Fmax bottleneck, so increase the II.
      [[intel::initiation_interval(5)]]
      do {
        // read in another byte if we have space for it
        if (bbs.HasSpaceForByte()) {
          bool read_valid;
          auto pd = InPipe::read(read_valid);

          if (read_valid) {
            unsigned char c = pd.data;
            done_reading = pd.flag;
            bbs.NewByte(c);
          }
        }

        if (decoding_next_symbol) {
          // decoding the next code symbol, so make sure we have enough bits to
          // do so 15 bits is the maximum bits to read both a symbol and the
          // extra run length bits (max 8 bits for the symbol, max 7 bits for
          // extra run length)
          if (bbs.Size() >= 15) {
            // read 15 bits
            ac_uint<15> next_bits = bbs.ReadUInt<15>();

            // find all possible dynamic run lengths
            // the symbol could be from 1 to 8 bits long and the number of extra
            // bits to read for the run length could be either 2, 3, or 7 bits
            // (3 possibilities in 'runlen_bits').
            [[intel::fpga_register]] ac_uint<7> extra_bit_vals[8][3];
            constexpr int runlen_bits[] = {2, 3, 7};
            #pragma unroll
            for (int out_codelen = 1; out_codelen <= 8; out_codelen++) {
              #pragma unroll
              for (int j = 0; j < 3; j++) {
                ac_uint<7> codebits_tmp(0);
                #pragma unroll
                for (int bit = 0; bit < runlen_bits[j]; bit++) {
                  codebits_tmp[bit] = next_bits[out_codelen + bit] & 0x1;
                }
                extra_bit_vals[out_codelen - 1][j] = codebits_tmp;
              }
            }

            // decode all possible code symbols from 1 to 8 bits
            ac_uint<8> codelencode_valid_bitmap(0);
            ac_uint<5> codelencode_offset[8];
            ac_uint<5> codelencode_base_idx[8];

            #pragma unroll
            for (int codelen = 1; codelen <= 8; codelen++) {
              ac_uint<8> codebits_tmp(0);
              #pragma unroll
              for (int bit = 0; bit < codelen; bit++) {
                codebits_tmp[codelen - bit - 1] = next_bits[bit] & 0x1;
              }
              unsigned char codebits = codebits_tmp;

              auto base_idx = codelencode_map_base_idx[codelen - 1];
              auto first_code = codelencode_map_first_code[codelen - 1];
              auto last_code = codelencode_map_last_code[codelen - 1];
              
              codelencode_base_idx[codelen - 1] = base_idx;
              codelencode_valid_bitmap[codelen - 1] =
                  ((codebits >= first_code) && (codebits < last_code)) ? 1 : 0;
              
              codelencode_offset[codelen - 1] = codebits - first_code;
            }

            // find the shortest matching code symbol
            ac_uint<3> shortest_match_len = ctz(codelencode_valid_bitmap) + 1;
            ac_uint<5> base_idx = codelencode_base_idx[shortest_match_len - 1];
            ac_uint<5> offset = codelencode_offset[shortest_match_len - 1];

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
              runlen = extra_bit_vals[shortest_match_len - 1][0] + 3;
              decoding_next_symbol = false;
              extend_symbol = codelens[codelens_idx-1];
              shift_amount = shortest_match_len + 2;
            } else if (symbol == 17) {
              // READ 3-BIT RUN LENGTH, ADD 3, AND EXTEND WITH 0's
              runlen = extra_bit_vals[shortest_match_len - 1][1] + 3;
              decoding_next_symbol = false;
              extend_symbol = 0;
              shift_amount = shortest_match_len + 3;
            } else if (symbol == 18) {
              // READ 7-BIT RUN LENGTH, ADD 11, AND EXTEND WITH 0's
              runlen = extra_bit_vals[shortest_match_len - 1][2] + 11;
              decoding_next_symbol = false;
              extend_symbol = 0;
              shift_amount = shortest_match_len + 7;
            }

            // shift the bit stream
            bbs.Shift(shift_amount);
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
      } while (codelens_idx < total_codes_second_table);

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

      // the first table is decoded, so now it is time to decode the second
      // table, which is actually two tables:
      //  literal table (symbols and lengths for the {length, distance} pair)
      //  distance table (the distances for the {length, distance} pair) 
      [[intel::fpga_register]] ac_uint<15> lit_map_first_code[15];
      [[intel::fpga_register]] ac_uint<15> lit_map_last_code[15];
      [[intel::fpga_register]] ac_uint<9> lit_map_base_idx[15];
      [[intel::fpga_register]] ac_uint<9> lit_map[286];

      [[intel::fpga_register]] ac_uint<15> dist_map_first_code[15];
      [[intel::fpga_register]] ac_uint<15> dist_map_last_code[15];
      [[intel::fpga_register]] ac_uint<5> dist_map_base_idx[15];
      [[intel::fpga_register]] ac_uint<5> dist_map[32];

      ac_uint<15> lit_map_next_code = 0;
      ac_uint<9> lit_map_counter = 0;
      for (unsigned char codelen = 1; codelen <= 15; codelen++) {
        lit_map_next_code <<= 1;
        lit_map_first_code[codelen - 1] = lit_map_next_code;
        lit_map_base_idx[codelen - 1] = lit_map_counter;
        for (unsigned short symbol = 0; symbol < numlitlencodes; symbol++) {
          auto inner_codelen = codelens[symbol];
          if (inner_codelen == codelen) {
            lit_map[lit_map_counter] = symbol;
            lit_map_counter++;
            lit_map_next_code++; 
          }
        }
        lit_map_last_code[codelen - 1] = lit_map_next_code;
      }

      ac_uint<15> dist_map_next_code = 0;
      ac_uint<5> dist_map_counter = 0;
      for (unsigned char codelen = 1; codelen <= 15; codelen++) {
        dist_map_next_code <<= 1;
        dist_map_first_code[codelen - 1] = dist_map_next_code;
        dist_map_base_idx[codelen - 1] = dist_map_counter;
        for (unsigned short symbol = 0; symbol < numdistcodes; symbol++) {
          auto inner_codelen = codelens[numlitlencodes + symbol];
          if (inner_codelen == codelen) {
            dist_map[dist_map_counter] = symbol;
            dist_map_counter++;
            dist_map_next_code++; 
          }
        }
        dist_map_last_code[codelen - 1] = dist_map_next_code;
      }

      // indicates whether we are reading a distance (or literal) currently
      bool reading_distance = false;

      // true is the stop code (256) has been decoded and the block is done
      bool stop_code_hit = false;

      // true when output is ready. the output will be either a character or
      // a length distance pair (see HuffmanData struct)
      bool out_ready = false;

      // the output data which is either a character or a length distance pair
      HuffmanData out_data;

      ac_uint<9> lit_symbol;
      ac_uint<5> dist_symbol;

      // main processing loop
#ifdef HUFFMAN_MAIN_LOOP_II
      [[intel::initiation_interval(HUFFMAN_MAIN_LOOP_II)]]
#endif
      do {
        // read in new data if the ByteBitStream has space for it and we aren't
        // done reading from the input pipe
        if (bbs.HasSpaceForByte()) {
          bool read_valid;
          auto pd = InPipe::read(read_valid);

          if (read_valid) {
            unsigned char c = pd.data;
            done_reading = pd.flag;
            bbs.NewByte(c);
          }
        }
        
        if (bbs.Size() >= 30) {
          // read the next 30 bits (we know we have them)
          ac_uint<30> next_bits = bbs.ReadUInt<30>();

          // find all possible dynamic lengths
          [[intel::fpga_register]] ac_uint<5> lit_extra_bit_vals[15][5];
          #pragma unroll
          for (int out_codelen = 1; out_codelen <= 15; out_codelen++) {
            #pragma unroll
            for (int in_codelen = 1; in_codelen <= 5; in_codelen++) {
              ac_uint<5> codebits_tmp(0);
              #pragma unroll
              for (int bit = 0; bit < in_codelen; bit++) {
                codebits_tmp[bit] = next_bits[out_codelen + bit] & 0x1;
              }
              lit_extra_bit_vals[out_codelen - 1][in_codelen - 1] = codebits_tmp;
            }
          }

          // find all possible dynamic distances
          [[intel::fpga_register]] ac_uint<15> dist_extra_bit_vals[15][15];
          #pragma unroll
          for (int out_codelen = 1; out_codelen <= 15; out_codelen++) {
            #pragma unroll
            for (int in_codelen = 1; in_codelen <= 15; in_codelen++) {
              ac_uint<15> codebits_tmp(0);
              #pragma unroll
              for (int bit = 0; bit < in_codelen; bit++) {
                codebits_tmp[bit] = next_bits[out_codelen + bit] & 0x1;
              }
              dist_extra_bit_vals[out_codelen - 1][in_codelen - 1] = codebits_tmp;
            }
          }

          // find all possible code lengths and offsets
          // TODO: get rid of selects here for literal vs distance symbol and just look stuff up in parallel
          // even though we write to every bit, we must initialize to 0
          // https://hsdes.intel.com/appstore/article/#/14015829976
          ac_uint<15> lit_codelen_valid_bitmap(0), dist_codelen_valid_bitmap(0);
          ac_uint<9> lit_codelen_offset[15], lit_codelen_base_idx[15];
          ac_uint<5> dist_codelen_offset[15], dist_codelen_base_idx[15];
          #pragma unroll
          for (unsigned char codelen = 1; codelen <= 15; codelen++) {
            ac_uint<15> codebits_tmp(0);
            #pragma unroll
            for (unsigned char bit = 0; bit < codelen; bit++) {
              codebits_tmp[codelen - bit - 1] = next_bits[bit] & 0x1;
            }
            unsigned short codebits = codebits_tmp;

            auto lit_base_idx = lit_map_base_idx[codelen - 1];
            auto lit_first_code = lit_map_first_code[codelen - 1];
            auto lit_last_code = lit_map_last_code[codelen - 1];
            auto dist_base_idx = dist_map_base_idx[codelen - 1];
            auto dist_first_code = dist_map_first_code[codelen - 1];
            auto dist_last_code = dist_map_last_code[codelen - 1];
            
            lit_codelen_valid_bitmap[codelen - 1] = ((codebits >= lit_first_code) && (codebits < lit_last_code)) ? 1 : 0;
            lit_codelen_base_idx[codelen - 1] = lit_base_idx;
            lit_codelen_offset[codelen - 1] = codebits - lit_first_code;

            dist_codelen_valid_bitmap[codelen - 1] = ((codebits >= dist_first_code) && (codebits < dist_last_code)) ? 1 : 0;
            dist_codelen_base_idx[codelen - 1] = dist_base_idx;
            dist_codelen_offset[codelen - 1] = codebits - dist_first_code;
          }

          // find the shortest matching length, which is the next decoded symbol
          auto lit_shortest_match_len = ctz(lit_codelen_valid_bitmap) + 1;
          auto dist_shortest_match_len = ctz(dist_codelen_valid_bitmap) + 1;

          // get the base index and offset based on the shortest match length
          auto lit_base_idx = lit_codelen_base_idx[lit_shortest_match_len - 1];
          auto lit_offset = lit_codelen_offset[lit_shortest_match_len - 1];
          auto dist_base_idx = dist_codelen_base_idx[dist_shortest_match_len - 1];
          auto dist_offset = dist_codelen_offset[dist_shortest_match_len - 1];
          ac_uint<9> lit_idx = lit_base_idx + lit_offset;
          ac_uint<9> dist_idx = dist_base_idx + dist_offset;

          // lookup the symbol using base_idx and offset
          lit_symbol = lit_map[lit_idx];
          dist_symbol =  dist_map[dist_idx];

          // we will either shift by shortest_match_len or by
          // shortest_match_len + num_extra_bits based on whether we read a
          // length, distance and/or its extra bits.
          // maximum value for shift_amount = 15 + 15 = 30
          ac_uint<5> shift_amount;

          if (!reading_distance) {
            shift_amount = lit_shortest_match_len;
            // currently parsing a symbol or length (same table)
            if (lit_symbol == 256) {
              // stop code hit, done this block
              stop_code_hit = true;
              out_ready = false;
            } else if (lit_symbol < 256) {
              // decoded a regular character
              out_data.len_or_sym = lit_symbol;
              out_data.dist_or_flag = -1;
              out_ready = true;
            } else if (lit_symbol <= 264) {
              // decoded a length with a static value
              out_data.len_or_sym = lit_symbol - ac_uint<9>(254);
              reading_distance = true;
            } else if (lit_symbol <= 284) {
              // decoded a length with a dynamic value
              ac_uint<3> num_extra_bits = (lit_symbol - ac_uint<9>(261)) >> 2;
              auto extra_bits_val = lit_extra_bit_vals[lit_shortest_match_len - 1][num_extra_bits - 1];
              out_data.len_or_sym = ((((lit_symbol - ac_uint<9>(265)) & 0x3) + ac_uint<3>(4)) << num_extra_bits) + ac_uint<2>(3) + extra_bits_val;
              shift_amount = lit_shortest_match_len + num_extra_bits;
              reading_distance = true;
            } else if (lit_symbol == 285) {
              // decoded a length with a static value
              out_data.len_or_sym = 258;
              reading_distance = true;
            } // else error, ignored
          } else {
            shift_amount = dist_shortest_match_len;
            // currently decoding a distance symbol
            if (dist_symbol <= 3) {
              // decoded a distance with a static value
              out_data.dist_or_flag = dist_symbol + 1;
            } else {
              // decoded a distance with a dynamic value
              // NOTE: should be <= 29, but not doing error checking
              ac_uint<4> num_extra_bits = (dist_symbol >> 1) - ac_uint<1>(1);
              auto extra_bits_val = dist_extra_bit_vals[dist_shortest_match_len - 1][num_extra_bits - 1];
              out_data.dist_or_flag = (((dist_symbol & 0x1) + ac_uint<2>(2)) << num_extra_bits) + ac_uint<1>(1) + extra_bits_val;
              shift_amount = dist_shortest_match_len + num_extra_bits;
            }
            out_ready = true;
            reading_distance = false;
          }

          // shift based on how many bits we read
          bbs.Shift(shift_amount);
        }

        // output data to downstream kernel if ready
        if (out_ready) {
          OutPipe::write(OutPipeBundleT(out_data));
          out_ready = false;
        }
      } while (!stop_code_hit);
    } while (!last_block);

    // notify downstream that we are done
    OutPipe::write(OutPipeBundleT(true));

    // read out the remaining data from the pipe
    // NOTE: don't really care about performance here
    while (!done_reading) {
      bool read_valid;
      auto pd = InPipe::read(read_valid);
      done_reading = pd.flag && read_valid;
    }
  });
}

template<typename InPipe, typename OutPipe>
event SubmitHeaderKernel(queue& q, int in_count, HeaderData* hdr_data_ptr, int* crc_ptr, int* out_count_ptr) {
  return q.single_task<HeaderKernelID>([=]() [[intel::kernel_args_restrict]] {
    using OutPipeBundleT = FlagBundle<unsigned char>;
    // read magic number (2 bytes)
    // read compression method (1 byte)
    // read 'flags' (1 byte)
    // read time (4 bytes)
    // read extra flags (1 byte)
    // read OS (1 byte)
    // read based on flags:
    //    if flags & 0x01 != 0: Flag = Text
    //    if flags & 0x04 != 0: Flag = Errata, read 2 bytes for 'length', read 'length' more bytes
    //    if flags & 0x08 != 0: Filename, read nullterminated string
    //    if flags & 0x02 != 0: CRC-16, read 2 bytes
    //    if flags & 0x10 != 0: Comment, read nullterminated string
    int i = 0;
    bool i_in_range = 0 < in_count;
    bool i_next_in_range = 1 < in_count;
    short state_counter = 0;
    short errata_len = 0;
    unsigned char curr_byte;
    HeaderState state = MagicNumber;
    device_ptr<HeaderData> hdr_data(hdr_data_ptr);

    unsigned char header_magic[2];
    unsigned char header_compression_method;
    unsigned char header_flags;
    unsigned char header_time[4];
    unsigned char header_os;
    unsigned char header_filename[256];
    unsigned char header_crc[2];
    header_filename[0] = '\0';

    device_ptr<int> crc(crc_ptr);
    device_ptr<int> out_count(out_count_ptr);

    // NOTE: this loop is not the main processing loop and therefore is
    // not critical (low trip count). However, the compiler doesn't know that
    // and tries to optimize for throughput (~Fmax/II). However, we don't want
    // this loop to be our Fmax bottleneck, so increase the II.
    [[intel::initiation_interval(4)]]
    while (state != SteadyState) {
      curr_byte = InPipe::read();

      switch (state) {
        case MagicNumber: {
          header_magic[state_counter] = curr_byte;
          state_counter++;
          if (state_counter == 2) {
            state = CompressionMethod;
            state_counter = 0;
          }
          break;
        }
        case CompressionMethod: {
          header_compression_method = curr_byte;
          state = Flags;
          break;
        }
        case Flags: {
          header_flags = curr_byte;
          state = Time;
          break;
        }
        case Time: {
          header_time[state_counter] = curr_byte;
          state_counter++;
          if (state_counter == 4) {
            state = ExtraFlags;
            state_counter = 0;
          }
          break;
        }
        case ExtraFlags: {
          state = OS;
          break;
        }
        case OS: {
          header_os = curr_byte;
          if (header_flags & 0x04) {
            state = Errata;
          } else if(header_flags & 0x08) {
            state = Filename;
          } else if(header_flags & 0x02) {
            state = CRC;
          } else if(header_flags & 0x10) {
            state = Comment;
          } else {
            state = SteadyState;
          }
          break;
        }
        case Errata: {
          if (state_counter == 0) {
            errata_len |= curr_byte;
            state_counter++;
          } else if (state_counter == 1) {
            errata_len |= (curr_byte << 8);
            state_counter++;
          } else {
            if ((state_counter - 2) == errata_len) {
              if(header_flags & 0x08) {
                state = Filename;
              } else if(header_flags & 0x02) {
                state = CRC;
              } else if(header_flags & 0x10) {
                state = Comment;
              } else {
                state = SteadyState;
              }
              state_counter = 0;
            } else {
              state_counter++;
            }
          }
          break;
        }
        case Filename: {
          header_filename[state_counter] = curr_byte;
          if (curr_byte == '\0') {
            if(header_flags & 0x02) {
              state = CRC;
            } else if(header_flags & 0x10) {
              state = Comment;
            } else {
              state = SteadyState;
            }
            state_counter = 0;
          } else {
            state_counter++;
          }
          break;
        }
        case CRC: {
          if (state_counter == 0) {
            header_crc[0] = curr_byte;
            state_counter++;
          } else if (state_counter == 1) {
            header_crc[1] = curr_byte;
            state_counter++;
          } else {
            if(header_flags & 0x10) {
              state = Comment;
            } else {
              state = SteadyState;
            }
            state_counter = 0;
          }
          break;
        }
        case Comment: {
          if (curr_byte == '\0') {
            state = SteadyState;
            state_counter = 0;
          } else {
            state_counter++;
          }
          break;
        }
        default: {
          break;
        }
      }

      i_in_range = i_next_in_range;
      i_next_in_range = i < (in_count - 2);
      i++;
    }

    unsigned char crc_bytes[4];
    unsigned char size_bytes[4];

    while (i_in_range) {
      bool valid_pipe_read;
      curr_byte = InPipe::read(valid_pipe_read);

      if (valid_pipe_read) {
        int tmp = (in_count - i - 1);
        if (tmp < 8) {
          if (tmp < 4) {
            size_bytes[3 - tmp] = curr_byte;
          } else {
            crc_bytes[7 - tmp] = curr_byte;
          }
        }
        OutPipe::write(OutPipeBundleT(curr_byte, (i == (in_count-1))));

        i_in_range = i_next_in_range;
        i_next_in_range = i < (in_count - 2);
        i++;
      }
    }

    unsigned int crc_local = 0, size_local = 0;
    for (int i = 0; i < 4; i++) {
      crc_local |= (unsigned int)(crc_bytes[i]) << (i*8);
      size_local |= (unsigned int)(size_bytes[i]) << (i*8);
    }

    // construct header data
    HeaderData hdr_data_loc;
    hdr_data_loc.magic[0] = header_magic[0];
    hdr_data_loc.magic[1] = header_magic[1];
    hdr_data_loc.compression_method = header_compression_method;
    hdr_data_loc.flags = header_flags;
    for(int i = 0; i < 4; i++) hdr_data_loc.time[i] = header_time[i];
    hdr_data_loc.os = header_os;
    for(int i = 0; i < 256; i++) hdr_data_loc.filename[i] = header_filename[i];
    hdr_data_loc.crc[0] = header_crc[0];
    hdr_data_loc.crc[1] = header_crc[1];

    // write back header data, crc, and size
    *hdr_data = hdr_data_loc;
    *crc = crc_local;
    *out_count = size_local;
  });
}

template<typename InPipe, typename OutPipe, unsigned literals_per_cycle>
std::vector<event> SubmitDecompressKernels(queue& q, int in_count,
                                           HeaderData *hdr_data_out,
                                           int *crc_out, int *size_out) {
  using HeaderToHuffmanPipe =
    ext::intel::pipe<HeaderToHuffmanPipeID, FlagBundle<unsigned char>>;
  using HuffmanToLZ77Pipe =
    ext::intel::pipe<HuffmanToLZ77PipeID, FlagBundle<HuffmanData>, 16>;  // TODO: experiment with depth
  using LZ77ToLiteralStackerPipe =
    ext::intel::pipe<LZ77ToLiteralStackerPipeID, FlagBundle<LiteralPack<literals_per_cycle>>>;

  auto header_event = SubmitHeaderKernel<InPipe, HeaderToHuffmanPipe>(q, in_count, hdr_data_out, crc_out, size_out);
  auto huffman_event = SubmitHuffmanDecoderKernel<HeaderToHuffmanPipe, HuffmanToLZ77Pipe>(q);
  auto lz77_event = SubmitLZ77DecoderKernel<HuffmanToLZ77Pipe, LZ77ToLiteralStackerPipe, literals_per_cycle>(q);
  auto lit_stacker_event = SubmitLiteralStackerKernel<LZ77ToLiteralStackerPipe, OutPipe, literals_per_cycle>(q);

  return {header_event, huffman_event, lz77_event, lit_stacker_event};
}

#endif /* __DECOMPRESSOR_HPP__ */