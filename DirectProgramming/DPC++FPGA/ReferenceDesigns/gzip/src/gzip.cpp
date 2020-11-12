// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

#include <CL/sycl.hpp>
#include <chrono>
#include <fstream>
#include <string>

#include "CompareGzip.hpp"
#include "WriteGzip.hpp"
#include "crc32.hpp"
#include "gzipkernel.hpp"
#include "kernels.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// The minimum file size of a file to be compressed.
// Any filesize less than this results in an error.
constexpr int minimum_filesize = kVec + 1;

bool help = false;

int CompressFile(queue &q, std::string &input_file, std::vector<std::string> outfilenames,
                 int iterations, bool report);

void Help(void) {
  // Command line arguments.
  // gzip [options] filetozip [options]
  // -h,--help                    : help

  // future options?
  // -p,performance : output perf metrics
  // -m,maxmapping=#  : maximum mapping size

  std::cout << "gzip filename [options]\n";
  std::cout << "  -h,--help                                : this help text\n";
  std::cout
      << "  -o=<filename>,--output-file=<filename>   : specify output file\n";
}

bool FindGetArg(std::string &arg, const char *str, int defaultval, int *val) {
  std::size_t found = arg.find(str, 0, strlen(str));
  if (found != std::string::npos) {
    int value = atoi(&arg.c_str()[strlen(str)]);
    *val = value;
    return true;
  }
  return false;
}

constexpr int kMaxStringLen = 40;

bool FindGetArgString(std::string &arg, const char *str, char *str_value,
                      size_t maxchars) {
  std::size_t found = arg.find(str, 0, strlen(str));
  if (found != std::string::npos) {
    const char *sptr = &arg.c_str()[strlen(str)];
    for (int i = 0; i < maxchars - 1; i++) {
      char ch = sptr[i];
      switch (ch) {
        case ' ':
        case '\t':
        case '\0':
          str_value[i] = 0;
          return true;
          break;
        default:
          str_value[i] = ch;
          break;
      }
    }
    return true;
  }
  return false;
}

size_t SyclGetExecTimeNs(event e) {
  size_t start_time =
      e.get_profiling_info<info::event_profiling::command_start>();
  size_t end_time =
      e.get_profiling_info<info::event_profiling::command_end>();
  return (end_time - start_time);
}

int main(int argc, char *argv[]) {
  std::string infilename = "";

  std::vector<std::string> outfilenames (kNumEngines);

  char str_buffer[kMaxStringLen] = {0};

  // Check the number of arguments specified
  if (argc != 3) {
    std::cerr << "Incorrect number of arguments. Correct usage: " << argv[0]
              << " <input-file> -o=<output-file>\n";
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      std::string sarg(argv[i]);
      if (std::string(argv[i]) == "-h") {
        help = true;
      }
      if (std::string(argv[i]) == "--help") {
        help = true;
      }

      FindGetArgString(sarg, "-o=", str_buffer, kMaxStringLen);
      FindGetArgString(sarg, "--output-file=", str_buffer, kMaxStringLen);
    } else {
      infilename = std::string(argv[i]);
    }
  }

  if (help) {
    Help();
    return 1;
  }

  try {
#ifdef FPGA_EMULATOR
    INTEL::fpga_emulator_selector device_selector;
#else
    INTEL::fpga_selector device_selector;
#endif
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(device_selector, dpc_common::exception_handler, prop_list);

    std::cout << "Running on device:  "
              << q.get_device().get_info<info::device::name>().c_str() << "\n";

    if (infilename == "") {
      std::cout << "Must specify a filename to compress\n\n";
      Help();
      return 1;
    }

    // next, check valid and acceptable parameter ranges.
    // if output filename not set, use the default
    // name, else use the name specified by the user
    outfilenames[0] = std::string(infilename) + ".gz";
    if (strlen(str_buffer)) {
      outfilenames[0] = std::string(str_buffer);
    }
    for (size_t i=1; i< kNumEngines; i++) {
      // Filenames will be of the form outfilename, outfilename2, outfilename3 etc.
      outfilenames[i] = outfilenames[0] + std::to_string(i+1);
    }

    std::cout << "Launching GZIP application with " << kNumEngines
              << " engines\n";

#ifdef FPGA_EMULATOR
    CompressFile(q, infilename, outfilenames, 1, true);
#else
    // warmup run - use this run to warmup accelerator. There are some steps in
    // the runtime that are only executed on the first kernel invocation but not
    // on subsequent invocations. So execute all that stuff here before we
    // measure performance (in the next call to CompressFile().
    CompressFile(q, infilename, outfilenames, 1, false);
    // profile performance
    CompressFile(q, infilename, outfilenames, 200, true);
#endif
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
  return 0;
}

struct KernelInfo {
  buffer<struct GzipOutInfo, 1> *gzip_out_buf;
  buffer<unsigned, 1> *current_crc;
  buffer<char, 1> *pobuf;
  buffer<char, 1> *pibuf;
  char *pobuf_decompress;

  uint32_t buffer_crc[kMinBufferSize];
  uint32_t refcrc;

  const char *pref_buffer;
  char *poutput_buffer;
  size_t file_size;
  struct GzipOutInfo out_info[kMinBufferSize];
  int iteration;
  bool last_block;
};

// returns 0 on success, otherwise a non-zero failure code.
int CompressFile(queue &q, std::string &input_file, std::vector<std::string> outfilenames,
                 int iterations, bool report) {
  size_t isz;
  char *pinbuf;

  // Read the input file
  std::string device_string =
      q.get_device().get_info<info::device::name>().c_str();

  // If
  // the device is S10, we pre-pin some buffers to
  // improve DMA performance, which is needed to
  // achieve peak kernel throughput. Pre-pinning is
  // only supported on the PAC-S10-USM BSP. It's not
  // needed on PAC-A10 to achieve peak performance.
  bool isS10 =  (device_string.find("s10") != std::string::npos);
  bool prepin = q.get_device().get_info<info::device::usm_host_allocations>();

  if (isS10 && !prepin) {
    std::cout << "Warning: Host allocations are not supported on this platform, which means that pre-pinning is not supported. DMA transfers may be slower than expected which may reduce application throughput.\n\n";
  }

  std::ifstream file(input_file,
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    isz = file.tellg();
    if (prepin) {
      pinbuf = (char *)malloc_host(
          isz, q.get_context());  // Pre-pin the buffer, for faster DMA
    } else {                      // throughput, using malloc_host().
      pinbuf = new char[isz];
    }
    file.seekg(0, std::ios::beg);
    file.read(pinbuf, isz);
    file.close();
  } else {
    std::cout << "Error: cannot read specified input file\n";
    return 1;
  }

  if (isz < minimum_filesize) {
    std::cout << "Minimum filesize for compression is " << minimum_filesize
              << "\n";
    return 1;
  }

  int buffers_count = iterations;

  // Create an array of kernel info structures and create buffers for kernel
  // input/output. The buffers are re-used between iterations, but enough 
  // disjoint buffers are created to support double-buffering.
  struct KernelInfo *kinfo[kNumEngines];
  for (size_t eng = 0; eng < kNumEngines; eng++) {
    kinfo[eng] =
        (struct KernelInfo *)malloc(sizeof(struct KernelInfo) * buffers_count);
    if (kinfo[eng] == NULL) {
      std::cout << "Cannot allocate kernel info buffer.\n";
      return 1;
    }
    for (int i = 0; i < buffers_count; i++) {
      kinfo[eng][i].file_size = isz;
      // Allocating slightly larger buffers (+ 16 * kVec) to account for
      // granularity of kernel writes
      int outputSize = kinfo[eng][i].file_size + 16 * kVec < kMinBufferSize
                           ? kMinBufferSize
                           : kinfo[eng][i].file_size + 16 * kVec;

      // Pre-pin buffer using malloc_host() to improve DMA bandwidth.
      if (i >= 3) {
        kinfo[eng][i].poutput_buffer = kinfo[eng][i - 3].poutput_buffer;
      } else {
        if (prepin) {
          kinfo[eng][i].poutput_buffer =
              (char *)malloc_host(outputSize, q.get_context());
        } else {
          kinfo[eng][i].poutput_buffer = (char *)malloc(outputSize);
        }
        if (kinfo[eng][i].poutput_buffer == NULL) {
          std::cout << "Cannot allocate output buffer.\n";
          free(kinfo);
          return 1;
        }
        // zero pages to fully allocate them
        memset(kinfo[eng][i].poutput_buffer, 0, outputSize);
      }

      kinfo[eng][i].last_block = true;
      kinfo[eng][i].iteration = i;
      kinfo[eng][i].pref_buffer = pinbuf;

      kinfo[eng][i].gzip_out_buf =
          i >= 3 ? kinfo[eng][i - 3].gzip_out_buf
                 : new buffer<struct GzipOutInfo, 1>(kMinBufferSize);
      kinfo[eng][i].current_crc = i >= 3
                                      ? kinfo[eng][i - 3].current_crc
                                      : new buffer<unsigned, 1>(kMinBufferSize);
      kinfo[eng][i].pibuf = i >= 3
                                ? kinfo[eng][i - 3].pibuf
                                : new buffer<char, 1>(kinfo[eng][i].file_size);
      kinfo[eng][i].pobuf =
          i >= 3 ? kinfo[eng][i - 3].pobuf : new buffer<char, 1>(outputSize);
      kinfo[eng][i].pobuf_decompress = (char *)malloc(kinfo[eng][i].file_size);
    }
  }

  // Create events for the various parts of the execution so that we can profile
  // their performance.
  event e_input_dma     [kNumEngines][buffers_count]; // Input to the GZIP engine. This is a transfer from host to device.
  event e_output_dma    [kNumEngines][buffers_count]; // Output from the GZIP engine. This is transfer from device to host.
  event e_crc_dma       [kNumEngines][buffers_count]; // Transfer CRC from device to host
  event e_size_dma      [kNumEngines][buffers_count]; // Transfer compressed file size from device to host
  event e_k_crc         [kNumEngines][buffers_count]; // CRC kernel
  event e_k_lz          [kNumEngines][buffers_count]; // LZ77 kernel
  event e_k_huff        [kNumEngines][buffers_count]; // Huffman Encoding kernel

#ifndef FPGA_EMULATOR
  dpc_common::TimeInterval perf_timer;
#endif

  
  /*************************************************/
  /* Main loop where the actual execution happens  */
  /*************************************************/
  for (int i = 0; i < buffers_count; i++) {
    for (size_t eng = 0; eng < kNumEngines; eng++) {
      // Transfer the input data, to be compressed, from host to device.
      e_input_dma[eng][i] = q.submit([&](handler &h) {
        auto in_data =
            kinfo[eng][i].pibuf->get_access<access::mode::discard_write>(h);
        h.copy(kinfo[eng][i].pref_buffer, in_data);
      });

      /************************************/
      /************************************/
      /*         LAUNCH GZIP ENGINE       */
      /************************************/
      /************************************/
      SubmitGzipTasks(q, kinfo[eng][i].file_size, kinfo[eng][i].pibuf,
                      kinfo[eng][i].pobuf, kinfo[eng][i].gzip_out_buf,
                      kinfo[eng][i].current_crc, kinfo[eng][i].last_block,
                      e_k_crc[eng][i], e_k_lz[eng][i], e_k_huff[eng][i], eng);

      // Transfer the output (compressed) data from device to host.
      e_output_dma[eng][i] = q.submit([&](handler &h) {
        auto out_data = kinfo[eng][i].pobuf->get_access<access::mode::read>(h);
        h.copy(out_data, kinfo[eng][i].poutput_buffer);
      });

      // Transfer the file size of the compressed output file from device to host.
      e_size_dma[eng][i] = q.submit([&](handler &h) {
        auto out_data =
            kinfo[eng][i].gzip_out_buf->get_access<access::mode::read>(h);
        h.copy(out_data, kinfo[eng][i].out_info);
      });

      // Transfer the CRC of the compressed output file from device to host.
      e_crc_dma[eng][i] = q.submit([&](handler &h) {
        auto out_data =
            kinfo[eng][i].current_crc->get_access<access::mode::read>(h);
        h.copy(out_data, kinfo[eng][i].buffer_crc);
      });
    }
  }

  // Wait for all kernels to complete
  for (int eng = 0; eng < kNumEngines; eng++) {
    for (int i = 0; i < buffers_count; i++) {
      e_output_dma[eng][i].wait();
      e_size_dma[eng][i].wait();
      e_crc_dma[eng][i].wait();
    }
  }

// Stop the timer.
#ifndef FPGA_EMULATOR
  double diff_total = perf_timer.Elapsed();
  double gbps = iterations * isz / (double)diff_total / 1000000000.0;
#endif

  // Check the compressed file size from each iteration. Make sure the size is actually
  // less-than-or-equal to the input size. Also calculate the remaining CRC.
  size_t compressed_sz[kNumEngines];
  for (int eng = 0; eng < kNumEngines; eng++) {
    compressed_sz[eng] = 0;
    for (int i = 0; i < buffers_count; i++) {
      if (kinfo[eng][i].out_info[0].compression_sz > kinfo[eng][i].file_size) {
        std::cerr << "Unsupported: compressed file larger than input file( "
                  << kinfo[eng][i].out_info[0].compression_sz << " )\n";
        return 1;
      }
      // The majority of the CRC is calculated by the CRC kernel on the FPGA. But the kernel
      // operates on quantized chunks of input data, so any remaining input data, that falls
      // outside the quanta, is included in the overall CRC calculation via the following 
      // function that runs on the host. The last argument is the running CRC that was computed
      // on the FPGA.
      kinfo[eng][i].buffer_crc[0] =
          Crc32(kinfo[eng][i].pref_buffer, kinfo[eng][i].file_size,
                kinfo[eng][i].buffer_crc[0]);
      // Accumulate the compressed size across all iterations. Used to 
      // compute compression ratio later.
      compressed_sz[eng] += kinfo[eng][i].out_info[0].compression_sz;
    }
  }

  // delete the file mapping now that all kernels are complete, and we've
  // snapped the time delta
  if (prepin) {
    free(pinbuf, q.get_context());
  } else {
    delete pinbuf;
  }

  // Write the output compressed data from the first iteration of each engine, to a file.
  for (int eng = 0; eng < kNumEngines; eng++) {
    // WriteBlockGzip() returns 1 on failure
    if (report && WriteBlockGzip(input_file, outfilenames[eng], kinfo[eng][0].poutput_buffer,
                        kinfo[eng][0].out_info[0].compression_sz,
                        kinfo[eng][0].file_size, kinfo[eng][0].buffer_crc[0])) {
      std::cout << "FAILED\n";
      return 1;
    }        
  }

  // Decompress the output from engine-0 and compare against the input file. Only engine-0's
  // output is verified since all engines are fed the same input data.
  if (report && CompareGzipFiles(input_file, outfilenames[0])) {
    std::cout << "FAILED\n";
    return 1;
  }

  // Generate throughput report
  // First gather all the execution times.
  size_t time_k_crc[kNumEngines];
  size_t time_k_lz[kNumEngines];
  size_t time_k_huff[kNumEngines];
  size_t time_input_dma[kNumEngines];
  size_t time_output_dma[kNumEngines];
  for (int eng = 0; eng < kNumEngines; eng++) {
    time_k_crc[eng] = 0;
    time_k_lz[eng] = 0;
    time_k_huff[eng] = 0;
    time_input_dma[eng] = 0;
    time_output_dma[eng] = 0;
    for (int i = 0; i < buffers_count; i++) {
      e_k_crc[eng][i].wait();
      e_k_lz[eng][i].wait();
      e_k_huff[eng][i].wait();
      time_k_crc[eng]       += SyclGetExecTimeNs(e_k_crc[eng][i]);
      time_k_lz[eng]        += SyclGetExecTimeNs(e_k_lz[eng][i]);
      time_k_huff[eng]      += SyclGetExecTimeNs(e_k_huff[eng][i]);
      time_input_dma[eng]   += SyclGetExecTimeNs(e_input_dma[eng][i]);
      time_output_dma[eng]  += SyclGetExecTimeNs(e_output_dma[eng][i]);
    }
  }

  if (report) {
    double compression_ratio =
        (double)((double)compressed_sz[0] / (double)isz / iterations);
#ifndef FPGA_EMULATOR
    std::cout << "Throughput: " << kNumEngines * gbps << " GB/s\n\n";
    for (int eng = 0; eng < kNumEngines; eng++) {
      std::cout << "TP breakdown for engine #" << eng << " (GB/s)\n";
      std::cout << "CRC = " << iterations * isz / (double)time_k_crc[eng]
                << "\n";
      std::cout << "LZ77 = " << iterations * isz / (double)time_k_lz[eng]
                << "\n";
      std::cout << "Huffman Encoding = "
                << iterations * isz / (double)time_k_huff[eng] << "\n";
      std::cout << "DMA host-to-device = "
                << iterations * isz / (double)time_input_dma[eng] << "\n";
      std::cout << "DMA device-to-host = "
                << iterations * isz / (double)time_output_dma[eng] << "\n\n";
    }
#endif
    std::cout << "Compression Ratio " << compression_ratio * 100 << "%\n";
  }

  // Cleanup anything that was allocated by this routine.
  for (int eng = 0; eng < kNumEngines; eng++) {
    for (int i = 0; i < buffers_count; i++) {
      if (i < 3) {
        delete kinfo[eng][i].gzip_out_buf;
        delete kinfo[eng][i].current_crc;
        delete kinfo[eng][i].pibuf;
        delete kinfo[eng][i].pobuf;
        if (prepin) {
          free(kinfo[eng][i].poutput_buffer, q.get_context());
        } else {
          free(kinfo[eng][i].poutput_buffer);
        }
      }
      free(kinfo[eng][i].pobuf_decompress);
    }
    free(kinfo[eng]);
  }

  if (report) std::cout << "PASSED\n";
  return 0;
}
