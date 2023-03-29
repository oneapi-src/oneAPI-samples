#ifndef __GZIP_METADATA_READER_HPP__
#define __GZIP_METADATA_READER_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "../common/common.hpp"
#include "constexpr_math.hpp"  // included from ../../../../include
#include "gzip_header_data.hpp"
#include "metaprogramming_utils.hpp"  // included from ../../../../include

//
// A kernel that streams in bytes of the GZIP file, strips away (and parses) the
// GZIP header and footer, and streams out the uncompressed data.
// The output of this kernel is a stream of DEFLATE formatted blocks.
//
//  Template parameters:
//    InPipe: a SYCL pipe that streams in compressed GZIP data, 1 byte at a time
//    OutPipe: a SYCL pipe that streams out the compressed GZIP data, 1 byte at
//      a time excluding the GZIP header and footer data
//
//  Arguments:
//    in_count: the number of compressed bytes
//    hdr_data: the parsed GZIP header
//    crc: the parsed CRC from the GZIP footer
//    out_count: the parsed uncompressed size from the GZIP footer
//
template <typename InPipe, typename OutPipe>
void GzipMetadataReader(int in_count, GzipHeaderData& hdr_data, int& crc,
                        int& out_count) {
  // ensure the InPipe and OutPipe are SYCL pipes
  static_assert(fpga_tools::is_sycl_pipe_v<InPipe>);
  static_assert(fpga_tools::is_sycl_pipe_v<OutPipe>);

  // the input and output pipe data types
  using InPipeBundleT = decltype(InPipe::read());
  using OutPipeBundleT = decltype(OutPipe::read());

  // make sure the input and output types are correct
  static_assert(std::is_same_v<InPipeBundleT, ByteSet<1>>);
  static_assert(std::is_same_v<OutPipeBundleT, FlagBundle<ByteSet<1>>>);

  /*
  GZIP FILE FORMAT:

  ===== HEADER =====
      2 bytes: magic number (0x1f8b)
      1 byte: compression method
      1 byte: 'flags'
      4 bytes: time
      1 byte: extra flags
      1 byte: OS
    Read more bytes based on flags:
      if flags & 0x01 != 0: Flag = Text
      if flags & 0x04 != 0: Flag = Errata, read 2 bytes for 'length',
                                   read 'length' more bytes
      if flags & 0x08 != 0: Filename, read nullterminated string
      if flags & 0x02 != 0: CRC-16, read 2 bytes
      if flags & 0x10 != 0: Comment, read nullterminated string

  ===== DATA =====
    1 or more consecutive DEFLATE compressed blocks

  ===== FOOTER =====
    4 bytes: CRC-32 Checksum
    4 bytes: Uncompressed data size in bytes
  */

  // This kernel reads the entire file (HEADER, DATA, and FOOTER) from the input
  // SYCL pipe 'InPipe, strips away and parses the HEADER and FOOTER, and
  // forwards the DATA to the next kernel through the SYCL pipe 'OutPipe'

  int i = 0;
  bool i_in_range = 0 < in_count;
  bool i_next_in_range = 1 < in_count;
  short state_counter = 0;
  short errata_len = 0;
  unsigned char curr_byte;
  GzipHeaderState state = GzipHeaderState::MagicNumber;

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
  [[intel::initiation_interval(4)]]  // NO-FORMAT: Attribute
  while (state != GzipHeaderState::SteadyState) {
    auto pipe_data = InPipe::read();
    curr_byte = pipe_data[0];

    // FSM for parsing the GZIP header, 1 byte at a time.
    switch (state) {
      case GzipHeaderState::MagicNumber: {
        header_magic[state_counter] = curr_byte;
        state_counter++;
        if (state_counter == 2) {
          state = GzipHeaderState::CompressionMethod;
          state_counter = 0;
        }
        break;
      }
      case GzipHeaderState::CompressionMethod: {
        header_compression_method = curr_byte;
        state = GzipHeaderState::Flags;
        break;
      }
      case GzipHeaderState::Flags: {
        header_flags = curr_byte;
        state = GzipHeaderState::Time;
        break;
      }
      case GzipHeaderState::Time: {
        header_time[state_counter] = curr_byte;
        state_counter++;
        if (state_counter == 4) {
          state = GzipHeaderState::ExtraFlags;
          state_counter = 0;
        }
        break;
      }
      case GzipHeaderState::ExtraFlags: {
        state = GzipHeaderState::OS;
        break;
      }
      case GzipHeaderState::OS: {
        header_os = curr_byte;
        if (header_flags & 0x04) {
          state = GzipHeaderState::Errata;
        } else if (header_flags & 0x08) {
          state = GzipHeaderState::Filename;
        } else if (header_flags & 0x02) {
          state = GzipHeaderState::CRC;
        } else if (header_flags & 0x10) {
          state = GzipHeaderState::Comment;
        } else {
          state = GzipHeaderState::SteadyState;
        }
        break;
      }
      case GzipHeaderState::Errata: {
        if (state_counter == 0) {
          errata_len |= curr_byte;
          state_counter++;
        } else if (state_counter == 1) {
          errata_len |= (curr_byte << 8);
          state_counter++;
        } else {
          if ((state_counter - 2) == errata_len) {
            if (header_flags & 0x08) {
              state = GzipHeaderState::Filename;
            } else if (header_flags & 0x02) {
              state = GzipHeaderState::CRC;
            } else if (header_flags & 0x10) {
              state = GzipHeaderState::Comment;
            } else {
              state = GzipHeaderState::SteadyState;
            }
            state_counter = 0;
          } else {
            state_counter++;
          }
        }
        break;
      }
      case GzipHeaderState::Filename: {
        header_filename[state_counter] = curr_byte;
        if (curr_byte == '\0') {
          if (header_flags & 0x02) {
            state = GzipHeaderState::CRC;
          } else if (header_flags & 0x10) {
            state = GzipHeaderState::Comment;
          } else {
            state = GzipHeaderState::SteadyState;
          }
          state_counter = 0;
        } else {
          state_counter++;
        }
        break;
      }
      case GzipHeaderState::CRC: {
        if (state_counter == 0) {
          header_crc[0] = curr_byte;
          state_counter++;
        } else if (state_counter == 1) {
          header_crc[1] = curr_byte;
          state_counter++;
        } else {
          if (header_flags & 0x10) {
            state = GzipHeaderState::Comment;
          } else {
            state = GzipHeaderState::SteadyState;
          }
          state_counter = 0;
        }
        break;
      }
      case GzipHeaderState::Comment: {
        if (curr_byte == '\0') {
          state = GzipHeaderState::SteadyState;
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
    auto pipe_data = InPipe::read(valid_pipe_read);
    curr_byte = pipe_data[0];

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
      OutPipe::write(OutPipeBundleT(pipe_data, (i == (in_count - 1))));

      i_in_range = i_next_in_range;
      i_next_in_range = i < (in_count - 2);
      i++;
    }
  }

  // parsing the GZIP footer
  // construct the 32-bit CRC and size (out_count) from the last 8 bytes read
  crc = 0;
  out_count = 0;
  for (int i = 0; i < 4; i++) {
    crc |= (unsigned int)(crc_bytes[i]) << (i * 8);
    out_count |= (unsigned int)(size_bytes[i]) << (i * 8);
  }

  // construct the header data
  hdr_data.magic[0] = header_magic[0];
  hdr_data.magic[1] = header_magic[1];
  hdr_data.compression_method = header_compression_method;
  hdr_data.flags = header_flags;
  for (int i = 0; i < 4; i++) hdr_data.time[i] = header_time[i];
  hdr_data.os = header_os;
  for (int i = 0; i < 256; i++) hdr_data.filename[i] = header_filename[i];
  hdr_data.crc[0] = header_crc[0];
  hdr_data.crc[1] = header_crc[1];
}

//
// Creates a kernel from the GZIP metadata reader function
//
template <typename Id, typename InPipe, typename OutPipe>
sycl::event SubmitGzipMetadataReader(sycl::queue& q, int in_count,
                                     GzipHeaderData* hdr_data_ptr, int* crc_ptr,
                                     int* out_count_ptr) {
  return q.single_task<Id>([=]() [[intel::kernel_args_restrict]] {

#if defined (IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer
    // lives on the device.
    // Knowing this, the compiler won't generate hardware to
    // potentially get data from the host.
    sycl::device_ptr<GzipHeaderData> hdr_data(hdr_data_ptr);
    sycl::device_ptr<int> crc(crc_ptr);
    sycl::device_ptr<int> out_count(out_count_ptr);
#else
    // Device pointers are not supported when targeting an FPGA 
    // family/part
    GzipHeaderData* hdr_data(hdr_data_ptr);
    int* crc(crc_ptr);
    int* out_count(out_count_ptr);
#endif    

    // local copies of the output data
    GzipHeaderData hdr_data_loc;
    int crc_loc;
    int out_count_loc;

    GzipMetadataReader<InPipe, OutPipe>(in_count, hdr_data_loc, crc_loc,
                                        out_count_loc);

    // write back the local copies of the output data
    *hdr_data = hdr_data_loc;
    *crc = crc_loc;
    *out_count = out_count_loc;
  });
}

#endif /* __GZIP_METADATA_READER_HPP__ */