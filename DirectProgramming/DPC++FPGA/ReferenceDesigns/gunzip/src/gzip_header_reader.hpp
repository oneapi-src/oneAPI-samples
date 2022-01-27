#ifndef __GZIP_HEADER_READER_HPP__
#define __GZIP_HEADER_READER_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "common.hpp"
#include "gzip_header_data.hpp"

using namespace sycl;

template<typename InPipe, typename OutPipe>
void GzipHeaderReader(int in_count, GzipHeaderData& hdr_data, int& crc,
                      int& out_count) {
  // the data type streamed out
  using OutPipeBundleT = FlagBundle<unsigned char>;

  // Format of the GZIP file header
  // 2 bytes: magic number (0x
  // 1 byte: compression method
  // 1 byte: 'flags'
  // 4 bytes: time
  // 1 byte: extra flags
  // 1 byte: OS
  // read more bytes based on flags:
  //    if flags & 0x01 != 0: Flag = Text
  //    if flags & 0x04 != 0: Flag = Errata, read 2 bytes for 'length',
  //                          read 'length' more bytes
  //    if flags & 0x08 != 0: Filename, read nullterminated string
  //    if flags & 0x02 != 0: CRC-16, read 2 bytes
  //    if flags & 0x10 != 0: Comment, read nullterminated string
  int i = 0;
  bool i_in_range = 0 < in_count;
  bool i_next_in_range = 1 < in_count;
  short state_counter = 0;
  short errata_len = 0;
  unsigned char curr_byte;
  GzipHeaderState state = MagicNumber;

  unsigned char header_magic[2];
  unsigned char header_compression_method;
  unsigned char header_flags;
  unsigned char header_time[4];
  unsigned char header_os;
  unsigned char header_filename[256];
  unsigned char header_crc[2];
  header_filename[0] = '\0';

  // NOTE: this loop is not the main processing loop and therefore is
  // not critical (low trip count). However, the compiler doesn't know that
  // and tries to optimize for throughput (~Fmax/II). However, we don't want
  // this loop to be our Fmax bottleneck, so increase the II.
  [[intel::initiation_interval(4)]]
  while (state != SteadyState) {
    curr_byte = InPipe::read();

    // FSM for parsing the GZIP header. We will process 1 byte per cycle here.
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

  // the last 8 bytes of the stream are the CRC and size (in bytes) of the file
  // this data will be sent back to the host to help validate the output
  unsigned char crc_bytes[4];
  unsigned char size_bytes[4];

  // finished reading the header, so now stream the bytes into the decompressor.
  // keep track of the last 8 bytes, which are the crc and output size. 
  // NOTE: we DO care about the performance of this loop, because it will feed
  // the rest of the decompressor.
  while (i_in_range) {
    bool valid_pipe_read;
    curr_byte = InPipe::read(valid_pipe_read);

    if (valid_pipe_read) {
      // keep track of the last 8 bytes
      int remaining_bytes = (in_count - i - 1);
      if (remaining_bytes < 8) {
        if (remaining_bytes < 4) {
          size_bytes[3 - remaining_bytes] = curr_byte;
        } else {
          crc_bytes[7 - remaining_bytes] = curr_byte;
        }
      }
      OutPipe::write(OutPipeBundleT(curr_byte, (i == (in_count-1))));

      i_in_range = i_next_in_range;
      i_next_in_range = i < (in_count - 2);
      i++;
    }
  }

  // construct the 32-bit CRC and size (out_count) from the last 8 bytes read
  crc = 0;
  out_count = 0;
  for (int i = 0; i < 4; i++) {
    crc |= (unsigned int)(crc_bytes[i]) << (i*8);
    out_count |= (unsigned int)(size_bytes[i]) << (i*8);
  }

  // construct the header data
  hdr_data.magic[0] = header_magic[0];
  hdr_data.magic[1] = header_magic[1];
  hdr_data.compression_method = header_compression_method;
  hdr_data.flags = header_flags;
  for(int i = 0; i < 4; i++) hdr_data.time[i] = header_time[i];
  hdr_data.os = header_os;
  for(int i = 0; i < 256; i++) hdr_data.filename[i] = header_filename[i];
  hdr_data.crc[0] = header_crc[0];
  hdr_data.crc[1] = header_crc[1];
}

template<typename Id, typename InPipe, typename OutPipe>
event SubmitGzipHeaderReader(queue& q, int in_count,
                             GzipHeaderData* hdr_data_ptr, int* crc_ptr,
                             int* out_count_ptr) {
  return q.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
    device_ptr<GzipHeaderData> hdr_data(hdr_data_ptr);
    device_ptr<int> crc(crc_ptr);
    device_ptr<int> out_count(out_count_ptr);

    // local copies of the output data
    GzipHeaderData hdr_data_loc;
    int crc_loc;
    int out_count_loc;
    
    GzipHeaderReader<InPipe, OutPipe>(in_count, hdr_data_loc, crc_loc,
                                      out_count_loc);

    // write back the local copies of the output data
    *hdr_data = hdr_data_loc;
    *crc = crc_loc;
    *out_count = out_count_loc;
  });
}

#endif /* __GZIP_HEADER_READER_HPP__ */