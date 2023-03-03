#ifndef __SNAPPY_DECOMPRESSOR_HPP__
#define __SNAPPY_DECOMPRESSOR_HPP__

#include <sycl/sycl.hpp>
#include <chrono>
#include <optional>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "../common/byte_stacker.hpp"
#include "../common/common.hpp"
#include "../common/lz77_decoder.hpp"
#include "constexpr_math.hpp"         // included from ../../../../include
#include "metaprogramming_utils.hpp"  // included from ../../../../include
#include "snappy_reader.hpp"

// declare the kernel and pipe names globally to reduce name mangling
class SnappyReaderKernelID;
class LZ77DecoderKernelID;
class ByteStackerKernelID;

class SnappyReaderToLZ77PipeID;
class LZ77ToByteStackerPipeID;

//
// Submits the kernels for the Snappy decompression engine and returns a list of
// SYCL events from each kernel launch.
//
// Template parameters:
//    InPipe: the input pipe that streams in compressed data,
//      'literals_per_cycle' byte at a time
//    OutPipe: the output pipe that streams out decompressed data,
//      'literals_per_cycle' at a time
//    literals_per_cycle: the number of literals streamed out the output stream.
//      This sets how many literals can be read from the input stream at once,
//      as well as the number that can be read at once from the history buffer
//      in the LZ77 decoder.
//
//  Arguments:
//    q: the SYCL queue
//    in_count: the number of compressed bytes
//    preamble_count: an output buffer for the uncompressed size read in the
//      Snappy preamble
//
template <typename InPipe, typename OutPipe, unsigned literals_per_cycle>
std::vector<sycl::event> SubmitSnappyDecompressKernels(
    sycl::queue& q, unsigned in_count, unsigned* preamble_count) {
  // check that the input and output pipe types are actually pipes
  static_assert(fpga_tools::is_sycl_pipe_v<InPipe>);
  static_assert(fpga_tools::is_sycl_pipe_v<OutPipe>);

  // 'literals_per_cycle' must be greater than 0 and a power of 2
  static_assert(literals_per_cycle > 0);
  static_assert(fpga_tools::IsPow2(literals_per_cycle));

  // the inter-kernel pipes for the snappy decompression engine
  constexpr int SnappyReaderToLZ77PipeDepth = 16;
  using SnappyReaderToLZ77Pipe = sycl::ext::intel::pipe<
      SnappyReaderToLZ77PipeID,
      FlagBundle<SnappyLZ77InputData<literals_per_cycle>>,
      SnappyReaderToLZ77PipeDepth>;

  auto snappy_reader_event =
      SubmitSnappyReader<SnappyReaderKernelID, InPipe, SnappyReaderToLZ77Pipe,
                         literals_per_cycle>(q, in_count, preamble_count);

  // the design only needs a ByteStacker kernel when literals_per_cycle > 1
  if constexpr (literals_per_cycle > 1) {
    using LZ77ToByteStackerPipe =
        sycl::ext::intel::pipe<LZ77ToByteStackerPipeID,
                               FlagBundle<BytePack<literals_per_cycle>>>;

    auto lz77_event =
        SubmitLZ77Decoder<LZ77DecoderKernelID, SnappyReaderToLZ77Pipe,
                          LZ77ToByteStackerPipe, literals_per_cycle,
                          kSnappyMaxLZ77Distance, kSnappyMaxLZ77Length>(q);
    auto byte_stacker_event =
        SubmitByteStacker<ByteStackerKernelID, LZ77ToByteStackerPipe, OutPipe,
                          literals_per_cycle>(q);

    return {snappy_reader_event, lz77_event, byte_stacker_event};
  } else {
    auto lz77_event =
        SubmitLZ77Decoder<LZ77DecoderKernelID, SnappyReaderToLZ77Pipe, OutPipe,
                          literals_per_cycle, kSnappyMaxLZ77Distance,
                          kSnappyMaxLZ77Length>(q);
    return {snappy_reader_event, lz77_event};
  }
}

// declare kernel and pipe names at the global scope to reduce name mangling
class ProducerId;
class ConsumerId;
class InPipeId;
class OutPipeId;

// the input and output pipe
using InPipe = sycl::ext::intel::pipe<InPipeId, ByteSet<kLiteralsPerCycle>>;
using OutPipe =
    sycl::ext::intel::pipe<OutPipeId, FlagBundle<BytePack<kLiteralsPerCycle>>>;

//
// The SNAPPY decompressor. See ../common/common.hpp for more information.
//
template <unsigned literals_per_cycle>
class SnappyDecompressor : public DecompressorBase {
 public:
  std::optional<std::vector<unsigned char>> DecompressBytes(
      sycl::queue& q, std::vector<unsigned char>& in_bytes, int runs,
      bool print_stats) {
    bool passed = true;
    unsigned in_count = in_bytes.size();
    int in_count_padded =
        fpga_tools::RoundUpToMultiple(in_count, kLiteralsPerCycle);

    // read the expected output size from the start of the file
    // this is used to size the output buffer
    unsigned out_count = 0;
    unsigned byte_idx = 0;
    unsigned shift = 0;
    bool keep_reading_preamble = true;
    while (keep_reading_preamble) {
      if (byte_idx > 4) {
        std::cerr << "ERROR: uncompressed length should not span more than 5"
                  << " bytes\n";
        std::terminate();
      }
      auto b = in_bytes[byte_idx];
      keep_reading_preamble = (b >> 7) & 0x1;
      out_count |= (b & 0x7F) << shift;
      shift += 7;
      byte_idx += 1;
    }

    std::vector<unsigned char> out_bytes(out_count);

    // round up the output count to the nearest multiple of kLiteralsPerCycle
    // this allows to ignore predicating the last writes to the output
    int out_count_padded =
        fpga_tools::RoundUpToMultiple(out_count, kLiteralsPerCycle);

    // host variables for output from device
    unsigned preamble_count_host;

    // track timing information in ms
    std::vector<double> time(runs);

    // input and output data pointers on the device using USM device allocations
    unsigned char *in, *out;
    unsigned* preamble_count;

    try {
#if defined (IS_BSP)
      // allocate memory on the device for the input and output
      if ((in = sycl::malloc_device<unsigned char>(in_count_padded, q)) ==
          nullptr) {
        std::cerr << "ERROR: could not allocate space for 'in'\n";
        std::terminate();
      }
      if ((out = sycl::malloc_device<unsigned char>(out_count_padded, q)) ==
          nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out'\n";
        std::terminate();
      }
      if ((preamble_count = sycl::malloc_device<unsigned>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'preamble_count'\n";
        std::terminate();
      }
#else
      // allocate shared memory
      if ((in = sycl::malloc_shared<unsigned char>(in_count_padded, q)) ==
          nullptr) {
        std::cerr << "ERROR: could not allocate space for 'in'\n";
        std::terminate();
      }
      if ((out = sycl::malloc_shared<unsigned char>(out_count_padded, q)) ==
          nullptr) {
        std::cerr << "ERROR: could not allocate space for 'out'\n";
        std::terminate();
      }
      if ((preamble_count = sycl::malloc_shared<unsigned>(1, q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'preamble_count'\n";
        std::terminate();
      }
#endif

      // copy the input data to the device memory and wait for the copy to
      // finish
      q.memcpy(in, in_bytes.data(), in_count * sizeof(unsigned char)).wait();

      // run the design multiple times to increase the accuracy of the timing
      for (int i = 0; i < runs; i++) {
        std::cout << "Launching kernels for run " << i << std::endl;

        // run the producer and consumer kernels
        auto producer_event =
            SubmitProducer<ProducerId, InPipe, literals_per_cycle>(
                q, in_count_padded, in);
        auto consumer_event =
            SubmitConsumer<ConsumerId, OutPipe, literals_per_cycle>(
                q, out_count_padded, out);

        // run the decompression kernels
        auto snappy_decompress_events =
            SubmitSnappyDecompressKernels<InPipe, OutPipe, kLiteralsPerCycle>(
                q, in_count, preamble_count);

        // wait for the producer and consumer to finish
        auto s = std::chrono::high_resolution_clock::now();
        producer_event.wait();
        consumer_event.wait();
        auto e = std::chrono::high_resolution_clock::now();

        // wait for the decompression kernels to finish
        for (auto& e : snappy_decompress_events) {
          e.wait();
        }

        std::cout << "All kernels finished for run " << i << std::endl;

        // calculate the time the kernels ran for, in milliseconds
        time[i] = std::chrono::duration<double, std::milli>(e - s).count();

        // Copy the output back from the device
        q.memcpy(out_bytes.data(), out, out_count * sizeof(unsigned char))
            .wait();
        q.memcpy(&preamble_count_host, preamble_count, sizeof(int)).wait();

        // validating the output
        // check the number of bytes we read
        if (preamble_count_host != out_count) {
          std::cerr << "ERROR: Out counts do not match: " << preamble_count_host
                    << " != " << out_count
                    << " (preamble_count_host != out_count)\n";
          passed = false;
        }
      }
    } catch (sycl::exception const& e) {
      std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
      std::terminate();
    }

    // free the allocated device memory
    sycl::free(in, q);
    sycl::free(out, q);
    sycl::free(preamble_count, q);

    // print the performance results
    if (passed && print_stats) {
      // NOTE: when run in emulation, these results do not accurately represent
      // the performance of the kernels on real FPGA hardware
      double avg_time_ms;
      if (runs > 1) {
        avg_time_ms =
            std::accumulate(time.begin() + 1, time.end(), 0.0) / (runs - 1);
      } else {
        avg_time_ms = time[0];
      }

      double compression_ratio =
          (double)(preamble_count_host) / (double)(in_count);

      // the number of input and output megabytes, respectively
      size_t out_mb = preamble_count_host * sizeof(unsigned char) * 1e-6;

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

#endif /* __SNAPPY_DECOMPRESSOR_HPP__ */
