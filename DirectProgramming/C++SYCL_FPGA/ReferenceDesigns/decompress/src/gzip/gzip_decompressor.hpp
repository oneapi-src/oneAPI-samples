#ifndef __GZIP_DECOMPRESSOR_HPP__
#define __GZIP_DECOMPRESSOR_HPP__

#include <sycl/sycl.hpp>
#include <chrono>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "../common/byte_stacker.hpp"
#include "../common/common.hpp"
#include "../common/lz77_decoder.hpp"
#include "../common/simple_crc32.hpp"
#include "constexpr_math.hpp"  // included from ../../../../include
#include "gzip_metadata_reader.hpp"
#include "huffman_decoder.hpp"
#include "metaprogramming_utils.hpp"  // included from ../../../../include

// declare the kernel and pipe names globally to reduce name mangling
class GzipMetadataReaderKernelID;
class HuffmanDecoderKernelID;
class LZ77DecoderKernelID;
class ByteStackerKernelID;

class GzipMetadataToHuffmanPipeID;
class HuffmanToLZ77PipeID;
class LZ77ToByteStackerPipeID;

// the depth of the pipe between the Huffman decoder and the LZ77 decoder.
// adding some extra depth here helps add some elasticity so that the Huffman
// decoder can computing while the LZ77 kernel reads from the history buffer
constexpr int kHuffmanToLZ77PipeDepth = 64;

//
// Submits the kernels for the GZIP decompression engine and returns a list of
// SYCL events from each kernel launch.
//
// Template parameters:
//    InPipe: the input pipe that streams in compressed data, 1 byte at a time
//    OutPipe: the output pipe that streams out decompressed data,
//      'literals_per_cycle' at a time
//    literals_per_cycle: the maximum number of literals written to the output
//      stream every cycle. This sets how many literals can be read from the
//      LZ77 history buffer at once.
//
//  Arguments:
//    q: the SYCL queue
//    in_count: the number of compressed bytes
//    hdr_data_out: a output buffer for the GZIP header data
//    crc_out: an output buffer for the CRC in the GZIP footer
//    count_out: an output buffer for the uncompressed size in the GZIP footer
//
template <typename InPipe, typename OutPipe, unsigned literals_per_cycle>
std::vector<sycl::event> SubmitGzipDecompressKernels(
    sycl::queue &q, int in_count, GzipHeaderData *hdr_data_out, int *crc_out,
    int *count_out) {
  // check that the input and output pipe types are actually pipes
  static_assert(fpga_tools::is_sycl_pipe_v<InPipe>);
  static_assert(fpga_tools::is_sycl_pipe_v<OutPipe>);

  // 'literals_per_cycle' must be greater than 0 and a power of 2
  static_assert(literals_per_cycle > 0);
  static_assert(fpga_tools::IsPow2(literals_per_cycle));

  // the inter-kernel pipes for the GZIP decompression engine
  using GzipMetadataToHuffmanPipe =
      sycl::ext::intel::pipe<GzipMetadataToHuffmanPipeID,
                             FlagBundle<ByteSet<1>>>;
  using HuffmanToLZ77Pipe =
      sycl::ext::intel::pipe<HuffmanToLZ77PipeID, FlagBundle<GzipLZ77InputData>,
                             kHuffmanToLZ77PipeDepth>;

  // submit the GZIP decompression kernels
  auto header_event =
      SubmitGzipMetadataReader<GzipMetadataReaderKernelID, InPipe,
                               GzipMetadataToHuffmanPipe>(
          q, in_count, hdr_data_out, crc_out, count_out);
  auto huffman_event =
      SubmitHuffmanDecoder<HuffmanDecoderKernelID, GzipMetadataToHuffmanPipe,
                           HuffmanToLZ77Pipe>(q);

  // the design only needs a ByteStacker kernel when literals_per_cycle > 1
  if constexpr (literals_per_cycle > 1) {
    using LZ77ToByteStackerPipe =
        sycl::ext::intel::pipe<LZ77ToByteStackerPipeID,
                               FlagBundle<BytePack<literals_per_cycle>>>;

    auto lz77_event =
        SubmitLZ77Decoder<LZ77DecoderKernelID, HuffmanToLZ77Pipe,
                          LZ77ToByteStackerPipe, literals_per_cycle,
                          kGzipMaxLZ77Distance, kGzipMaxLZ77Length>(q);
    auto byte_stacker_event =
        SubmitByteStacker<ByteStackerKernelID, LZ77ToByteStackerPipe, OutPipe,
                          literals_per_cycle>(q);

    return {header_event, huffman_event, lz77_event, byte_stacker_event};
  } else {
    auto lz77_event =
        SubmitLZ77Decoder<LZ77DecoderKernelID, HuffmanToLZ77Pipe, OutPipe,
                          literals_per_cycle, kGzipMaxLZ77Distance,
                          kGzipMaxLZ77Length>(q);
    return {header_event, huffman_event, lz77_event};
  }
}

// declare kernel and pipe names at the global scope to reduce name mangling
class ProducerId;
class ConsumerId;
class InPipeId;
class OutPipeId;

// the input and output pipe
using InPipe = sycl::ext::intel::pipe<InPipeId, ByteSet<1>>;
using OutPipe =
    sycl::ext::intel::pipe<OutPipeId, FlagBundle<BytePack<kLiteralsPerCycle>>>;

//
// The GZIP decompressor. See ../common/common.hpp for more information.
//
template <unsigned literals_per_cycle>
class GzipDecompressor : public DecompressorBase {
 public:
  std::optional<std::vector<unsigned char>> DecompressBytes(
      sycl::queue &q, std::vector<unsigned char> &in_bytes, int runs,
      bool print_stats) {
    int in_count = in_bytes.size();

    // read the expected output size from the last 4 bytes of the file
    std::vector<unsigned char> last_4_bytes(in_bytes.end() - 4, in_bytes.end());
    unsigned out_count = *(reinterpret_cast<unsigned *>(last_4_bytes.data()));
    std::vector<unsigned char> out_bytes(out_count);

    // round up the output count to the nearest multiple of literals_per_cycle,
    // which allows us to not predicate the last writes to the output buffer
    // from the device.
    int out_count_padded =
        fpga_tools::RoundUpToMultiple(out_count, literals_per_cycle);

    // the GZIP header data. This is parsed by the GZIPMetadataReader kernel
    GzipHeaderData hdr_data_h;

    // the GZIP footer data. This is parsed by the GZIPMetadataReader kernel.
    unsigned int crc_h, count_h;

    // track timing information in ms
    std::vector<double> time_ms(runs);

    // input and output data pointers on the device using USM device allocations
    unsigned char *in, *out;

    // the GZIP header data (see gzip_header_data.hpp)
    GzipHeaderData *hdr_data;

    // the GZIP footer data, where 'count' is the expected number of bytes
    // in the uncompressed file
    int *crc, *count;

    bool passed = true;

    try {
#if defined (IS_BSP)
      // allocate memory on the device
      if ((in = sycl::malloc_device<unsigned char>(in_count, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'in'\n";
        std::terminate();
      }
      if ((out = sycl::malloc_device<unsigned char>(out_count_padded, q)) ==
          nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out'\n";
        std::terminate();
      }
      if ((hdr_data = sycl::malloc_device<GzipHeaderData>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'hdr_data'\n";
        std::terminate();
      }
      if ((crc = sycl::malloc_device<int>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'crc'\n";
        std::terminate();
      }
      if ((count = sycl::malloc_device<int>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'count'\n";
        std::terminate();
      }
#else
      // allocate shared memory 
      if ((in = sycl::malloc_shared<unsigned char>(in_count, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'in'\n";
        std::terminate();
      }
      if ((out = sycl::malloc_shared<unsigned char>(out_count_padded, q)) ==
          nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out'\n";
        std::terminate();
      }
      if ((hdr_data = sycl::malloc_shared<GzipHeaderData>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'hdr_data'\n";
        std::terminate();
      }
      if ((crc = sycl::malloc_shared<int>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'crc'\n";
        std::terminate();
      }
      if ((count = sycl::malloc_shared<int>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'count'\n";
        std::terminate();
      }
#endif

      // copy the input data to the device memory and wait for the copy to
      // finish
      q.memcpy(in, in_bytes.data(), in_count * sizeof(unsigned char)).wait();

      // run the design multiple times to increase the accuracy of the timing
      for (int i = 0; i < runs; i++) {
        std::cout << "Launching kernels for run " << i << std::endl;

        auto producer_event =
            SubmitProducer<ProducerId, InPipe, 1>(q, in_count, in);
        auto consumer_event =
            SubmitConsumer<ConsumerId, OutPipe, literals_per_cycle>(
                q, out_count_padded, out);

        auto gzip_decompress_events =
            SubmitGzipDecompressKernels<InPipe, OutPipe, literals_per_cycle>(
                q, in_count, hdr_data, crc, count);

        auto s = std::chrono::high_resolution_clock::now();
        producer_event.wait();
        consumer_event.wait();
        auto e = std::chrono::high_resolution_clock::now();

        // wait for the decompression kernels to finish
        for (auto &e : gzip_decompress_events) {
          e.wait();
        }

        std::cout << "All kernels have finished for run " << i << std::endl;

        // duration in milliseconds
        time_ms[i] = std::chrono::duration<double, std::milli>(e - s).count();

        // Copy the output back from the device
        q.memcpy(out_bytes.data(), out, out_count * sizeof(unsigned char))
            .wait();
        q.memcpy(&hdr_data_h, hdr_data, sizeof(GzipHeaderData)).wait();
        q.memcpy(&crc_h, crc, sizeof(int)).wait();
        q.memcpy(&count_h, count, sizeof(int)).wait();

        // validating the output
        // check the magic header we read
        if (hdr_data_h.MagicNumber() != 0x1f8b) {
          auto save_flags = std::cerr.flags();
          std::cerr << "ERROR: Incorrect magic header value of 0x" << std::hex
                    << std::setw(4) << std::setfill('0')
                    << hdr_data_h.MagicNumber() << " (should be 0x1f8b)\n";
          std::cerr.flags(save_flags);
          passed = false;
        }

        // check the number of bytes we read
        if (count_h != out_count) {
          std::cerr << "ERROR: Out counts do not match: " << count_h
                    << " != " << out_count << "(count_h != out_count)\n";
          passed = false;
        }

        // compute the CRC of the output data
        auto crc32_out = SimpleCRC32(0, out_bytes.data(), out_count);

        // check that the computed CRC matches the expectation (crc_h is the
        // CRC-32 that is in the GZIP footer).
        if (crc32_out != crc_h) {
          auto save_flags = std::cout.flags();
          std::cerr << std::hex << std::setw(4) << std::setfill('0');
          std::cerr << "ERROR: output data CRC does not match the expected CRC "
                    << "0x" << crc32_out << " != 0x" << crc_h
                    << " (result != expected)\n";
          std::cout.flags(save_flags);
          passed = false;
        }
      }
    } catch (sycl::exception const &e) {
      std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
      std::terminate();
    }

    // free the allocated device memory
    sycl::free(in, q);
    sycl::free(out, q);
    sycl::free(hdr_data, q);
    sycl::free(crc, q);
    sycl::free(count, q);

    // print the performance results
    if (passed && print_stats) {
      // NOTE: when run in emulation, these results do not accurately represent
      // the performance of the kernels on real FPGA hardware
      double avg_time_ms;
      if (runs > 1) {
        avg_time_ms = std::accumulate(time_ms.begin() + 1, time_ms.end(), 0.0) /
                      (runs - 1);
      } else {
        avg_time_ms = time_ms[0];
      }

      double compression_ratio = (double)(count_h) / (double)(in_count);

      // the number of input and output megabytes, respectively
      size_t out_mb = count_h * sizeof(unsigned char) * 1e-6;

      std::cout << "Execution time: " << avg_time_ms << " ms\n";
      std::cout << "Output Throughput: " << (out_mb / (avg_time_ms * 1e-3))
                << " MB/s\n";
      std::cout << "Compression Ratio: " << compression_ratio << ":1"
                << "\n";
    }

    if (passed) {
      return out_bytes;
    } else {
      return {};
    }
  }
};

#endif /* __GZIP_DECOMPRESSOR_HPP__ */
