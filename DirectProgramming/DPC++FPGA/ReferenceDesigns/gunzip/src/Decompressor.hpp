#ifndef __DECOMPRESSOR_HPP__
#define __DECOMPRESSOR_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "ByteBitStream.hpp"
#include "HeaderData.hpp"

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

template<int bits> 
void PrintUACInt(ac_int<bits, false>& x) {
  for (int i = bits-1; i >= 0; i--) {
    PRINTF("%d", x[i] & 0x1);
  }
}

//
// TODO
//
template<typename T>
struct FlagBundle {
  FlagBundle() : data(T(0)), flag(false) {}
  FlagBundle(T d_in) : data(d_in), flag(false) {}
  FlagBundle(T d_in, bool f_in) : data(d_in), flag(f_in) {}

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

class HeaderKernelID;
class HuffmanDecoderKernelID;
class LZ77DecoderKernelID;

class HeaderToHuffmanPipeID;
class HuffmanToLZ77PipeID;
using HeaderToHuffmanPipe =
  ext::intel::pipe<HeaderToHuffmanPipeID, FlagBundle<unsigned char>>;
using HuffmanToLZ77Pipe =
  ext::intel::pipe<HuffmanToLZ77PipeID, FlagBundle<HuffmanData>>;

template<typename InPipe, typename OutPipe>
event SubmitLZ77DecoderKernel(queue& q) {
  return q.single_task<LZ77DecoderKernelID>([=] {
    unsigned idx = 0;
    constexpr unsigned kMaxHistory = 32768;
    constexpr unsigned kMaxHistoryMask = kMaxHistory - 1;

    static_assert(fpga_tools::IsPow2(kMaxHistory));
    unsigned char history[kMaxHistory];

    FlagBundle<HuffmanData> pipe_data;
    bool done;

    do {
      // TODO: Non-blocking?
      // TODO: make this a single loop, rather than nested loop in else statement?
      // TODO: history memory is bad :(
      pipe_data = InPipe::read();
      done = pipe_data.flag;

      if (!done) {
        if (pipe_data.data.dist_or_flag == -1) {
          unsigned char c = (unsigned char)(pipe_data.data.len_or_sym & 0xFF);
          history[idx] = c;
          idx = (idx + 1) & kMaxHistoryMask;
          //PRINTF("out: %d\n", (unsigned int)c);
          OutPipe::write(FlagBundle<unsigned char>(c));
        } else {
          unsigned short len = pipe_data.data.len_or_sym;
          unsigned short dist = pipe_data.data.dist_or_flag;

          //PRINTF("copy %d back %d\n", len, dist);

          unsigned int ridx = (idx - dist) & kMaxHistoryMask;
          [[intel::ivdep(history)]]
          for (int i = 0; i < len; i++) {
            unsigned char c = history[ridx];
            ridx = (ridx + 1) & kMaxHistoryMask;
            history[idx] = c;
            idx = (idx + 1) & kMaxHistoryMask;
            OutPipe::write(FlagBundle<unsigned char>(c));
          }
        }
      }
    } while (!done);
    
    OutPipe::write(FlagBundle<unsigned char>(0, true));
  });
}

template<typename InPipe, typename OutPipe>
event SubmitHuffmanDecoderKernel(queue& q) {
  return q.single_task<HuffmanDecoderKernelID>([=] {
    ByteBitStream bbs;
    bool last_block;
    bool done_reading = false;

    do {
      unsigned short type = 0xFF;
      unsigned short last_block_num = 0xFF;
      unsigned short numlitlencodes = 0, numdistcodes = 0, numcodelencodes = 0;
      bool parsing_first_table = true, parsing_second_table = true;
      unsigned short codelencodelen_count = 0;

      constexpr unsigned short codelencodelen_idxs[] =
        {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13};
      unsigned char codelencodelen[19] = {0};

      do {
        if (bbs.HasSpaceForByte()) {
          auto pd = InPipe::read();
          unsigned char c = pd.data;
          bbs.NewByte(c);
        }

        if (bbs.Size() >= 5) {
          if (last_block_num == 0xFF) {
            last_block_num = bbs.ReadUInt(1);
            bbs.Shift(1);
            last_block = (last_block_num == 1);
            //PRINTF("last_block_num = %hu\n", last_block_num);
          } else if (type == 0xFF) {
            type = bbs.ReadUInt(2);
            bbs.Shift(2);
            //PRINTF("type = %hu\n", type);
          } else if (numlitlencodes == 0) {
            numlitlencodes = bbs.ReadUInt(5) + (unsigned short)257;
            bbs.Shift(5);
            //PRINTF("numlitlencodes = %hu\n", numlitlencodes);
          } else if (numdistcodes == 0) {
            numdistcodes = bbs.ReadUInt(5) + (unsigned short)1;
            bbs.Shift(5);
            //PRINTF("numdistcodes = %hu\n", numdistcodes);
          } else if (numcodelencodes == 0) {
            numcodelencodes = bbs.ReadUInt(4) + (unsigned short)4;
            bbs.Shift(4);
            //PRINTF("numcodelencodes = %hu\n", numcodelencodes);
          } else if (codelencodelen_count < numcodelencodes) {
            unsigned short tmp = bbs.ReadUInt(3);
            bbs.Shift(3);
            //PRINTF("tmp = %hu\n", tmp);
            codelencodelen[codelencodelen_idxs[codelencodelen_count]] = tmp;
            codelencodelen_count++;
            parsing_first_table = (codelencodelen_count != numcodelencodes);
          }
        }
      } while (parsing_first_table);

      /*
      for (int i = 0; i < 19; i++) {
        PRINTF("%hu\n", codelencodelen[i]);
      }
      PRINTF("DONE codelencodelen\n");
      */

      int codelencode[1024];
      for (int i = 0; i < 1024; i++) { codelencode[i] = -1; }
      unsigned short next_code = 0;
      for (unsigned short codelen = 1; codelen < 8; codelen++) {
        next_code <<= 1;
        unsigned short start_bit = 1 << codelen;
        for (unsigned short symbol = 0; symbol < 19; symbol++) {
          unsigned short inner_codelen = codelencodelen[symbol];
          if (inner_codelen == codelen) {
            codelencode[start_bit | next_code] = symbol;
            //PRINTF("%u : %u\n", (start_bit | next_code), symbol);
            next_code++;
          }
        }
      }

      //PRINTF("DONE codelencode\n");

      // length of codelens is MAX(numlitlencodes + numdistcodes)
      // = MAX((2^5 + 257) + (2^5 + 1)) = 322
      int codelens[322];
      for (int i = 0; i < 322; i++) { codelens[i] = -1; }
      int tmp_symbol;
      unsigned short symbol;
      unsigned int codebits = 1;
      unsigned short runlenbits;
      unsigned short runlenoffset;
      bool reading_next_symbol = true;
      bool runlen_extend;
      unsigned short runlen;
      unsigned int codelens_idx = 0;
      int onecount = 0, otherpositivecount = 0;
      do {
        if (bbs.HasSpaceForByte()) {
          auto pd = InPipe::read();
          unsigned char c = pd.data;
          bbs.NewByte(c);
        }

        if (reading_next_symbol) {
          do {
            unsigned short next_bit = bbs.ReadUInt(1);
            bbs.Shift(1);
            codebits = (codebits << 1) | next_bit;
            tmp_symbol = codelencode[codebits];
            //PRINTF("codebits: %hu, tmp_symbol: %d\n", codebits, tmp_symbol);
            symbol = tmp_symbol;
          } while (tmp_symbol == -1 && !bbs.Empty());
          //PRINTF("SYMBOL: %hu\n", symbol);
        } else if (bbs.Size() >= runlenbits) {
          unsigned short tmp = bbs.ReadUInt(runlenbits);
          bbs.Shift(runlenbits);
          runlen = tmp + runlenoffset;
          //PRINTF("RUNLEN: %hu\n", runlen);
        }
        

        // symbol ready
        if (reading_next_symbol && tmp_symbol != -1) {
          //PRINTF("SYMBOL: %hu\n", symbol);
          if (symbol <= 15) {
            // ADD SYMBOL
            codelens[codelens_idx++] = symbol;
            reading_next_symbol = true;
            if (codelens_idx >= numlitlencodes) {
              if (symbol == 1) {
                onecount++;
              } else if (symbol > 0) {
                otherpositivecount++;
              }
            }
            //PRINTF("APPEND SYMBOL: %hu\n", symbol);
            symbol = -1;
          } else if (symbol == 16) {
            // READ 2-BIT RUN LENGTH, ADD 3, AND EXTEND LAST ELEMENT
            runlenbits = 2;
            runlenoffset = 3;
            reading_next_symbol = false;
            runlen_extend = true;
          } else if (symbol == 17) {
            // READ 3-BIT RUN LENGTH, ADD 3, AND EXTEND WITH 0's
            runlenbits = 3;
            runlenoffset = 3;
            reading_next_symbol = false;
            runlen_extend = false;
          } else if (symbol == 18) {
            // READ 7-BIT RUN LENGTH, ADD 11, AND EXTEND WITH 0's
            runlenbits = 7;
            runlenoffset = 11;
            reading_next_symbol = false;
            runlen_extend = false;
          }
          codebits = 1;
        } else {
          unsigned short extend_val = runlen_extend ? codelens[codelens_idx-1] : 0;
          for (int i = 0; i < runlen; i++) {
            codelens[codelens_idx++] = extend_val;
            if (codelens_idx >= numlitlencodes) {
              if (extend_val == 1) {
                onecount++;
              } else if (extend_val > 0) {
                otherpositivecount++;
              }
            }
          }
          reading_next_symbol = true;
          //PRINTF("EXTEND: %hu * %hu\n", extend_val, runlen);
        }

        parsing_second_table = (codelens_idx < (numlitlencodes + numdistcodes));
      } while (parsing_second_table);

      if (onecount == 1 && otherpositivecount == 0) {
        int extend_amount = 32 - numdistcodes;
        for (int i = 0; i < extend_amount; i++) {
          codelens[numlitlencodes + numdistcodes + i] = 0;
        }
        codelens[numlitlencodes + 31] = 1;
        numdistcodes += extend_amount;
      }

      /*
      for (int i = 0; i < (numlitlencodes + numdistcodes); i++) {
        PRINTF("%hu\n", codelens[i]);
      }
      PRINTF("len(codelens): %d\n", codelens_idx);
      PRINTF("DONE codelens\n");
      */

      // TODO: revisit 16384, affects the unrolled below for checking on match

      //int litlencode[16384];  // 9 bits is suffcient + sign bit = 10
      //int distcode[16384];  // 6 bits is sufficient + sign bit = 7
      short litlencode[16384];  // 9 bits is suffcient + sign bit = 10
      char distcode[16384];  // 6 bits is sufficient + sign bit = 7
      for (int i = 0; i < 16384; i++) { litlencode[i] = -1; }
      for (int i = 0; i < 16384; i++) { distcode[i] = -1; }

      // TODO: revisit 16
      next_code = 0;
      for (unsigned short codelen = 1; codelen < 16; codelen++) {
        next_code <<= 1;
        unsigned short start_bit = 1 << codelen;
        for (unsigned short symbol = 0; symbol < numlitlencodes; symbol++) {
          unsigned short inner_codelen = codelens[symbol];
          if (inner_codelen == codelen) {
            litlencode[start_bit | next_code] = symbol;
            //PRINTF("%u : %u\n", (start_bit | next_code), symbol);
            next_code++;
          }
        }
      }
      next_code = 0;
      for (unsigned short codelen = 1; codelen < 16; codelen++) {
        next_code <<= 1;
        unsigned short start_bit = 1 << codelen;
        for (unsigned short symbol = 0; symbol < numdistcodes; symbol++) {
          unsigned short inner_codelen = codelens[numlitlencodes + symbol];
          if (inner_codelen == codelen) {
            distcode[start_bit | next_code] = symbol;
            //PRINTF("%u : %u\n", (start_bit | next_code), symbol);
            next_code++;
          }
        }
      }

      /*
      PRINTF("litlencode:\n")
      for (int i = 0; i < 16384; i++) {
        if (litlencode[i] != -1) {
          PRINTF("%d : %hu\n", i, (unsigned int)litlencode[i]);
        }
      }
      */

      /*
      PRINTF("distcode:\n")
      for (int i = 0; i < 16384; i++) {
        if (distcode[i] != -1) {
          PRINTF("%d : %hu\n", i, (unsigned int)distcode[i]);
        }
      }
      */

      //PRINTF("DONE litlencode and distcode\n");
      bool stop_code_hit = false;
      bool out_ready = false;
      unsigned int num_extra_bits = 0;
      HuffmanData out_data;

      enum BlockParsingState {
        Symbol, ExtraRunLengthBits, DistanceSymbol, ExtraDistanceBits, Done
      };

      BlockParsingState state = Symbol;
      //codebits = 1;

      // TODO: really need improvement here
      do {
        //PRINTF("TOP\n");
        if (bbs.HasSpaceForByte() && !done_reading) {
          auto pd = InPipe::read();
          unsigned char c = pd.data;
          done_reading = pd.flag;
          bbs.NewByte(c);
        }

        if (bbs.Size() >= 13) {
        if (state == Symbol || state == DistanceSymbol) {
          constexpr unsigned short shift_amounts[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
          //constexpr unsigned short masks[] = {1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535};
          bool match[16];
          int symbols[16];
          ac_int<16, false> next_bits = bbs.ReadUInt(13);


          // TODO: splicing?
          #pragma unroll
          for (int i = 0; i < 13; i++) {
            unsigned short codebits = (1 << (i + 1));
            #pragma unroll
            for (int j = 0; j < i + 1; j++) {
              codebits |= (next_bits[j] & 0x1) << (i - j);
            }
            int sym = (state == Symbol) ? litlencode[codebits] : distcode[codebits];
            //PRINTF("codebits: %d\n", codebits);
            symbols[i] = sym;
            match[i] = (sym != -1);
          }

          unsigned short shift_amount;

          #pragma unroll
          for (int i = 12; i >= 0; i--) {
            if (match[i]) {
              tmp_symbol = symbols[i];
              symbol = tmp_symbol;
              shift_amount = shift_amounts[i];
            }
          }
          bbs.Shift(shift_amount);

          /*
          do {
            unsigned short next_bit = bbs.ReadUInt(1);
            bbs.Shift(1);
            codebits = (codebits << 1) | next_bit;
            PRINTF("codebits: %d\n", codebits);
            tmp_symbol = (state == Symbol) ? litlencode[codebits] : distcode[codebits];
            symbol = tmp_symbol;
          } while (tmp_symbol == -1 && !bbs.Empty());
          */


          if (tmp_symbol != -1) {
            //PRINTF("TMP: %d %d\n", tmp_symbol, codebits);
            if (symbol == 256) {
              stop_code_hit = true;
              out_ready = false;
              state = Done;
            } else if (state == Symbol) {
              if (symbol < 256) {
                //PRINTF("SYMBOL: %d\n", tmp_symbol);
                out_data.len_or_sym = symbol;
                out_data.dist_or_flag = -1;
                out_ready = true;
                state = Symbol;
              } else if (symbol <= 264) {
                //PRINTF("RUN LEN SYMBOL: %d\n", tmp_symbol);
                out_data.len_or_sym = symbol - 254;
                //PRINTF("STATIC RUN LEN: %d\n", out_data.len_or_sym);
                state = DistanceSymbol;
              } else if (symbol <= 284) {
                //PRINTF("RUN LEN SYMBOL: %d\n", tmp_symbol);
                num_extra_bits = (symbol - 261) / 4;
                state = ExtraRunLengthBits;
              } else if (symbol == 285) {
                //PRINTF("RUN LEN SYMBOL: %d\n", tmp_symbol);
                out_data.len_or_sym = 258;
                //PRINTF("STATIC RUN LEN: 258\n");
                state = DistanceSymbol;
              } // else error, ignored
            } else if (state == DistanceSymbol) {
              //PRINTF("DIST SYMBOL: %d\n", tmp_symbol);
              if (symbol <= 3) {
                out_data.dist_or_flag = symbol + 1;
                state = Symbol;
                out_ready = true;
                //PRINTF("STATIC DIST: %d\n", out_data.dist_or_flag);
              } else {
                // NOTE: should be <= 29, but not doing error checking
                num_extra_bits = (symbol / 2) - 1;
                state = ExtraDistanceBits;
              }
            }

            codebits = 1;
          }
        } else {
          if (bbs.Size() >= num_extra_bits) {
            unsigned short extra_bits = bbs.ReadUInt(num_extra_bits);
            bbs.Shift(num_extra_bits);

            if (state == ExtraRunLengthBits) {
              out_data.len_or_sym = (((symbol - 265) % 4 + 4) << num_extra_bits) + 3 + extra_bits;
              //PRINTF("DYNAMIC RUN LEN: %d\n", out_data.len_or_sym);
              state = DistanceSymbol;
            } else if (state == ExtraDistanceBits) {
              out_data.dist_or_flag = ((symbol % 2 + 2) << num_extra_bits) + 1 + extra_bits;
              //PRINTF("DYNAMIC DIST: %d\n", out_data.dist_or_flag);
              out_ready = true;
              state = Symbol;
            }
          }
        }
        }

        if (out_ready) {
          //PRINTF("%d %d\n", out_data.len_or_sym, out_data.dist_or_flag);
          OutPipe::write(FlagBundle<HuffmanData>(out_data));
          out_ready = false;
        }
      } while (!stop_code_hit);
      //PRINTF("stop_code_hit and done\n");
    } while (!last_block);

    // NOTE: don't really care about performance here
    while (!done_reading) {
      auto pd = InPipe::read();
      done_reading = pd.flag;
    }

    OutPipe::write(FlagBundle<HuffmanData>(HuffmanData(), true));
  });
}

template<typename InPipe, typename OutPipe>
event SubmitHeaderKernel(queue& q, HeaderData* hdr_data_ptr, int count, int* crc_ptr, int* size_ptr) {
  return q.single_task<HeaderKernelID>([=]() [[intel::kernel_args_restrict]] {
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
    short state_counter = 0;
    short errata_len = 0;
    unsigned char curr_byte;
    HeaderState state = MagicNumber;
    device_ptr<HeaderData> hdr_data(hdr_data_ptr);
    HeaderData hdr_data_loc;
    hdr_data_loc.filename[0] = '\0';

    device_ptr<int> crc(crc_ptr);
    device_ptr<int> size(size_ptr);

    while (state != SteadyState) {
      curr_byte = InPipe::read();

      switch (state) {
        case MagicNumber: {
          hdr_data_loc.magic[state_counter] = curr_byte;
          state_counter++;
          if (state_counter == 2) {
            state = CompressionMethod;
            state_counter = 0;
          }
          break;
        }
        case CompressionMethod: {
          hdr_data_loc.compression_method = curr_byte;
          state = Flags;
          break;
        }
        case Flags: {
          hdr_data_loc.flags = curr_byte;
          state = Time;
          break;
        }
        case Time: {
          hdr_data_loc.time[state_counter] = curr_byte;
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
          hdr_data_loc.os = curr_byte;
          if (hdr_data_loc.flags & 0x04) {
            state = Errata;
          } else if(hdr_data_loc.flags & 0x08) {
            state = Filename;
          } else if(hdr_data_loc.flags & 0x02) {
            state = CRC;
          } else if(hdr_data_loc.flags & 0x10) {
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
              if(hdr_data_loc.flags & 0x08) {
                state = Filename;
              } else if(hdr_data_loc.flags & 0x02) {
                state = CRC;
              } else if(hdr_data_loc.flags & 0x10) {
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
          hdr_data_loc.filename[state_counter] = curr_byte;
          if (curr_byte == '\0') {
            if(hdr_data_loc.flags & 0x02) {
              state = CRC;
            } else if(hdr_data_loc.flags & 0x10) {
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
          if (state_counter == 0 || state_counter == 1) {
            hdr_data_loc.crc[state_counter] = curr_byte;
            state_counter++;
          } else {
            if(hdr_data_loc.flags & 0x10) {
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

      i++;
    }

    unsigned char crc_bytes[4];
    unsigned char size_bytes[4];

    while (i < count) {
      curr_byte = InPipe::read();
      int tmp = (count - i - 1);
      if (tmp < 8) {
        if (tmp < 4) {
          size_bytes[3 - tmp] = curr_byte;
        } else {
          crc_bytes[7 - tmp] = curr_byte;
        }
      }
      OutPipe::write(FlagBundle<unsigned char>(curr_byte, (i == (count-1))));
      i++;
    }

    unsigned int crc_local = 0, size_local = 0;
    for (int i = 0; i < 4; i++) {
      crc_local |= (unsigned int)(crc_bytes[i]) << (i*8);
      size_local |= (unsigned int)(size_bytes[i]) << (i*8);
    }

    // write back header data, crc, and size
    *hdr_data = hdr_data_loc;
    *crc = crc_local;
    *size = size_local;
  });
}

#endif // __DECOMPRESSOR_HPP__