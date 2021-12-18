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

class USMDebugger {
public:
  void Init(queue& q, int count = 64) {
    count_ = count;
    if ((data_ = malloc_host<long long>(count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'data_'\n";
      std::terminate();
    }
    for (int i = 0; i < count; i++) {
      data_[i] = -1;
    }
  }

  void Destroy(queue& q) {
    sycl::free(data_, q);
    data_ = nullptr;
  }

  void Write(int idx, long long val) const {
    host_ptr<long long> data_host_ptr(data_);
    data_host_ptr[idx] = val;
  }
  
  void Print(int max_count=0) {
    std::cout << "===== USMDebugger dump =====\n";
    for (int i = 0; i < count_ && (max_count == 0 || i < max_count); i++) {
      std::cout << i << ": " << data_[i] << "\n";
    }
    std::cout << "============================\n";
  }

private:
  long long *data_;
  int count_;
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
      // TODO: Non-blocking pipe read?
      // TODO: make this a single loop, rather than nested loop in else statement?
      // TODO: history memory is bad :(
      pipe_data = InPipe::read();
      done = pipe_data.flag;

      if (!done) {
        if (pipe_data.data.dist_or_flag == -1) {
          unsigned char c = (unsigned char)(pipe_data.data.len_or_sym & 0xFF);
          history[idx] = c;
          idx = (idx + 1) & kMaxHistoryMask;
          OutPipe::write(FlagBundle<unsigned char>(c));
        } else {
          unsigned short len = pipe_data.data.len_or_sym;
          unsigned short dist = pipe_data.data.dist_or_flag;

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
  return q.single_task<HuffmanDecoderKernelID>([=]() [[intel::kernel_args_restrict]] {
    ByteBitStream bbs;
    bool last_block;
    bool done_reading = false;

    [[intel::disable_loop_pipelining]]  // ya?
    do {
      unsigned short type = 0xFF;
      unsigned short last_block_num = 0xFF;
      unsigned short numlitlencodes = 0, numdistcodes = 0, numcodelencodes = 0;
      bool parsing_first_table = true, parsing_second_table = true;
      unsigned short codelencodelen_count = 0;

      constexpr unsigned short codelencodelen_idxs[] =
        {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
      unsigned char codelencodelen[19] = {0};

      // NOTE: the compiler chose an II=2 here and sacrificed Fmax. However,
      // we know this loop will have a low trip count, so set the II higher
      // so we don't lower the Fmax with this non-critical loop.
      // Trial-and-error to arrive upon this II.
      [[intel::initiation_interval(4)]]
      do {
        if (bbs.HasSpaceForByte()) {
          auto pd = InPipe::read();
          unsigned char c = pd.data;
          bbs.NewByte(c);
        }

        if (bbs.Size() >= 5) {
          if (last_block_num == 0xFF) {
            last_block_num = bbs.ReadUInt(1);
            //PRINTF("last_block_num: %u\n", last_block_num);
            bbs.Shift(1);
            last_block = (last_block_num == 1);
          } else if (type == 0xFF) {
            type = bbs.ReadUInt(2);
            //PRINTF("type: %u\n", type);
            bbs.Shift(2);
          } else if (numlitlencodes == 0) {
            numlitlencodes = bbs.ReadUInt(5) + (unsigned short)257;
            //PRINTF("numlitlencodes: %u\n", numlitlencodes);
            bbs.Shift(5);
          } else if (numdistcodes == 0) {
            numdistcodes = bbs.ReadUInt(5) + (unsigned short)1;
            //PRINTF("numdistcodes: %u\n", numdistcodes);
            bbs.Shift(5);
          } else if (numcodelencodes == 0) {
            numcodelencodes = bbs.ReadUInt(4) + (unsigned short)4;
            //PRINTF("numcodelencodes: %u\n", numcodelencodes);
            bbs.Shift(4);
          } else if (codelencodelen_count < numcodelencodes) {
            unsigned short tmp = bbs.ReadUInt(3);
            bbs.Shift(3);
            codelencodelen[codelencodelen_idxs[codelencodelen_count]] = tmp;
            //PRINTF("codelencodelen[%u] = %u\n", codelencodelen_idxs[codelencodelen_count], tmp);
            codelencodelen_count++;
            parsing_first_table = (codelencodelen_count != numcodelencodes);
          }
        }
      } while (parsing_first_table);

      int codelencode[256];  // 1 << 8 = 256
      for (int i = 0; i < 256; i++) { codelencode[i] = -1; }
      unsigned short next_code = 0;
      for (unsigned short codelen = 1; codelen < 8; codelen++) {
        next_code <<= 1;
        unsigned short start_bit = 1 << codelen;
        for (unsigned short symbol = 0; symbol < 19; symbol++) {
          unsigned short inner_codelen = codelencodelen[symbol];
          if (inner_codelen == codelen) {
            codelencode[start_bit | next_code] = symbol;
            //PRINTF("(%u) %u : %u\n", codelen, (start_bit | next_code), symbol);
            next_code++;
          }
        }
      }

      // length of codelens is MAX(numlitlencodes + numdistcodes)
      // = MAX((2^5 + 257) + (2^5 + 1)) = 322
      int codelens[322];
      for (int i = 0; i < 322; i++) { codelens[i] = -1; }
      int tmp_symbol;
      unsigned short early_symbol;
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
            early_symbol = tmp_symbol;
          } while (tmp_symbol == -1 && !bbs.Empty());
        } else if (bbs.Size() >= runlenbits) {
          unsigned short tmp = bbs.ReadUInt(runlenbits);
          bbs.Shift(runlenbits);
          runlen = tmp + runlenoffset;
        }
        

        // symbol ready
        if (reading_next_symbol && tmp_symbol != -1) {
          if (early_symbol <= 15) {
            // ADD SYMBOL
            codelens[codelens_idx++] = early_symbol;
            reading_next_symbol = true;
            if (codelens_idx >= numlitlencodes) {
              if (early_symbol == 1) {
                onecount++;
              } else if (early_symbol > 0) {
                otherpositivecount++;
              }
            }
            early_symbol = -1;
          } else if (early_symbol == 16) {
            // READ 2-BIT RUN LENGTH, ADD 3, AND EXTEND LAST ELEMENT
            runlenbits = 2;
            runlenoffset = 3;
            reading_next_symbol = false;
            runlen_extend = true;
          } else if (early_symbol == 17) {
            // READ 3-BIT RUN LENGTH, ADD 3, AND EXTEND WITH 0's
            runlenbits = 3;
            runlenoffset = 3;
            reading_next_symbol = false;
            runlen_extend = false;
          } else if (early_symbol == 18) {
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

      // TODO reconsider types here, use ac_int for smaller widths
      [[intel::fpga_register]] unsigned short lit_map_first_code[15];
      [[intel::fpga_register]] unsigned short lit_map_last_code[15];
      [[intel::fpga_register]] ac_int<9, false> lit_map_base_idx[15];
      [[intel::fpga_register]] ac_int<9, false> lit_map[286];

      [[intel::fpga_register]] unsigned short dist_map_first_code[15];
      [[intel::fpga_register]] unsigned short dist_map_last_code[15];
      [[intel::fpga_register]] ac_int<5, false> dist_map_base_idx[15];
      [[intel::fpga_register]] ac_int<5, false> dist_map[32];

      unsigned short lit_map_next_code = 0;
      unsigned short lit_map_counter = 0;
      for (unsigned char codelen = 1; codelen <= 15; codelen++) {
        lit_map_next_code <<= 1;
        lit_map_first_code[codelen - 1] = lit_map_next_code;
        lit_map_base_idx[codelen - 1] = lit_map_counter;
        for (unsigned short symbol = 0; symbol < numlitlencodes; symbol++) {
          unsigned short inner_codelen = codelens[symbol];
          if (inner_codelen == codelen) {
            lit_map[lit_map_counter] = symbol;
            lit_map_counter++;
            lit_map_next_code++; 
          }
        }
        lit_map_last_code[codelen - 1] = lit_map_next_code;
      }

      unsigned short dist_map_next_code = 0;
      unsigned short dist_map_counter = 0;
      for (unsigned char codelen = 1; codelen <= 15; codelen++) {
        dist_map_next_code <<= 1;
        dist_map_first_code[codelen - 1] = dist_map_next_code;
        dist_map_base_idx[codelen - 1] = dist_map_counter;
        for (unsigned short symbol = 0; symbol < numdistcodes; symbol++) {
          unsigned short inner_codelen = codelens[numlitlencodes + symbol];
          if (inner_codelen == codelen) {
            dist_map[dist_map_counter] = symbol;
            dist_map_counter++;
            dist_map_next_code++; 
          }
        }
        dist_map_last_code[codelen - 1] = dist_map_next_code;
      }

      bool stop_code_hit = false;
      bool out_ready = false;
      unsigned int num_extra_bits = 0;
      HuffmanData out_data;

      // CONSIDERATION:
      //  enum lowest enum can go is 8 bits. Could use an ac_int<2, false> here
      //  but is it really worth it? Quartus likely sweeps these away.
      enum BlockParsingState : unsigned char {
        Symbol, ExtraRunLengthBits, DistanceSymbol, ExtraDistanceBits
      };
      BlockParsingState state = Symbol;

      // main processing loop
      ac_int<9, false> lit_symbol;
      ac_int<5, false> dist_symbol;
      do {
        // TODO: make this if then if, rather than if-elseif
        if (bbs.HasSpaceForByte() && !done_reading) {
          bool read_valid;
          auto pd = InPipe::read(read_valid);

          if (read_valid) {
            unsigned char c = pd.data;
            done_reading = pd.flag;
            bbs.NewByte(c);
          }
        }
        
        if (bbs.Size() >= 15) {
          if (state == Symbol || state == DistanceSymbol) {
            // read the next 15 bits (we know we have them)
            ac_int<15, false> next_bits = bbs.ReadUInt15();

            // find all possible code lengths and offsets
            bool codelen_valid_bitmap[15];  // TODO: use ac_int here?
            ac_int<9, false> codelen_offset[15];
            ac_int<9, false> codelen_base_idx[15];
            #pragma unroll
            for (unsigned char codelen = 1; codelen <= 15; codelen++) {
              ac_int<15, false> codebits_tmp(0);
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

              auto base_idx = (state == Symbol) ? (unsigned short)lit_base_idx
                                                : (unsigned short)dist_base_idx;
              auto first_code = (state == Symbol) ? lit_first_code
                                                  : dist_first_code;
              auto last_code = (state == Symbol) ? lit_last_code
                                                 : dist_last_code;
              
              codelen_base_idx[codelen - 1] = base_idx;
              codelen_valid_bitmap[codelen - 1] =
                  (codebits >= first_code) & (codebits < last_code);;
              
              codelen_offset[codelen - 1] = codebits - first_code;
            }

            unsigned char shortest_match_len;
            unsigned short base_idx;
            unsigned short offset;
            #pragma unroll
            for (unsigned char codelen = 15; codelen >= 1; codelen--) {
              if (codelen_valid_bitmap[codelen - 1]) {
                shortest_match_len = codelen;
                base_idx = codelen_base_idx[codelen - 1];
                offset = codelen_offset[codelen - 1];
              }
            }

            // lookup symbol using base_idx and offset
            lit_symbol = lit_map[base_idx + offset];
            dist_symbol =  dist_map[base_idx + offset];

            // shift by however many bits we matched
            bbs.Shift(shortest_match_len);

            // TODO: can we precompute the compares on symbol?
            if (state == Symbol) {
              if (lit_symbol == 256) {
                stop_code_hit = true;
                out_ready = false;
                state = ExtraDistanceBits;
              } else if (lit_symbol < 256) {
                out_data.len_or_sym = lit_symbol;
                out_data.dist_or_flag = -1;
                out_ready = true;
                state = Symbol;
              } else if (lit_symbol <= 264) {
                out_data.len_or_sym = lit_symbol - 254;
                state = DistanceSymbol;
              } else if (lit_symbol <= 284) {
                num_extra_bits = (lit_symbol - 261) / 4;
                state = ExtraRunLengthBits;
              } else if (lit_symbol == 285) {
                out_data.len_or_sym = 258;
                state = DistanceSymbol;
              } // else error, ignored
            } else if (state == DistanceSymbol) {
              if (dist_symbol <= 3) {
                out_data.dist_or_flag = dist_symbol + 1;
                state = Symbol;
                out_ready = true;
              } else {
                // NOTE: should be <= 29, but not doing error checking
                num_extra_bits = (dist_symbol / 2) - 1;
                state = ExtraDistanceBits;
              }
            }
          } else {
            if (bbs.Size() >= num_extra_bits) {
              unsigned short extra_bits = bbs.ReadUInt(num_extra_bits);
              bbs.Shift(num_extra_bits);

              if (state == ExtraRunLengthBits) {
                out_data.len_or_sym = (((lit_symbol - 265) % 4 + 4) << num_extra_bits) + 3 + extra_bits;
                state = DistanceSymbol;
              } else if (state == ExtraDistanceBits) {
                out_data.dist_or_flag = ((dist_symbol % 2 + 2) << num_extra_bits) + 1 + extra_bits;
                out_ready = true;
                state = Symbol;
              }
            }
          }
        }

        // output data to downstream kernel if ready
        if (out_ready) {
          OutPipe::write(FlagBundle<HuffmanData>(out_data));
          out_ready = false;
        }
      } while (!stop_code_hit);
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
    bool i_in_range = 0 < count;
    bool i_next_in_range = 1 < count;
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
    device_ptr<int> size(size_ptr);

    // NOTE: the compiler chose an II=2 here and sacrificed Fmax. However,
    // we know this loop will have a low trip count, so set the II higher
    // so we don't lower the Fmax with this non-critical loop.
    // Trial-and-error to arrive upon this II.
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
      i_next_in_range = i < (count - 2);
      i++;
    }

    unsigned char crc_bytes[4];
    unsigned char size_bytes[4];

    while (i_in_range) {
      bool valid_pipe_read;
      curr_byte = InPipe::read(valid_pipe_read);

      if (valid_pipe_read) {
        int tmp = (count - i - 1);
        if (tmp < 8) {
          if (tmp < 4) {
            size_bytes[3 - tmp] = curr_byte;
          } else {
            crc_bytes[7 - tmp] = curr_byte;
          }
        }
        OutPipe::write(FlagBundle<unsigned char>(curr_byte, (i == (count-1))));

        i_in_range = i_next_in_range;
        i_next_in_range = i < (count - 2);
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
    *size = size_local;
  });
}

#endif // __DECOMPRESSOR_HPP__