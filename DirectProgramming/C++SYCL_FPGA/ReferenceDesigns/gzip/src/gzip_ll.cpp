#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <chrono>
#include <fstream>
#include <string>

#include "CompareGzip.hpp"
#include "WriteGzip.hpp"
#include "crc32.hpp"
#include "gzipkernel_ll.hpp"
#include "kernels.hpp"

#include "exception_handler.hpp"


using namespace sycl;

// The minimum file size of a file to be compressed.
// Any filesize less than this results in an error.
constexpr int minimum_filesize = kVec + 1;

const int N_BUFFERING = 2;  // Number of sets of I/O buffers to
                            // allocate, for the purpose of overlapping kernel
                            // execution with buffer preparation.


bool help = false;

int CompressFile(queue &q, std::string &input_file,
                 std::vector<std::string> outfilenames, int iterations,
                 bool report);

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
  size_t end_time = e.get_profiling_info<info::event_profiling::command_end>();
  return (end_time - start_time);
}

int main(int argc, char *argv[]) {
  std::string infilename = "";

  std::vector<std::string> outfilenames(kNumEngines);

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

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(selector, fpga_tools::exception_handler, prop_list);

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<info::device::name>().c_str() 
              << std::endl;

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
    for (size_t i = 1; i < kNumEngines; i++) {
      // Filenames will be of the form outfilename, outfilename2, outfilename3
      // etc.
      outfilenames[i] = outfilenames[0] + std::to_string(i + 1);
    }

    std::cout << "Launching Low-Latency GZIP application with " << kNumEngines
              << " engines\n";

#ifdef FPGA_EMULATOR
    CompressFile(q, infilename, outfilenames, 10, true);
#elif FPGA_SIMULATOR
    CompressFile(q, infilename, outfilenames, 2, true);
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
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
      std::cerr << "If you are targeting the FPGA simulator, compile with "
                   "-DFPGA_SIMULATOR.\n";
    }
    std::terminate();
  }
  return 0;
}

struct KernelInfo {
  struct GzipOutInfo
      *gzip_out_buf;      // Contains meta data about the compressed file
  uint32_t *current_crc;  // Partial CRC of the input file

  char **pobuf_ptr_array;  // Array of pointers to input files to be compressed
  char **pibuf_ptr_array;  // Corresponding array of pointers to output from the
                           // gzip engine (the compressed results)

  char *pref_buffer;  // Original input copy of file to compress

  size_t input_size;
  size_t output_size;
  bool last_block;

  std::vector<event> kernel_event;  // Events for the execution of the kernels
                                    // that comprise the GZIP engine
};

// returns 0 on success, otherwise a non-zero failure code.
int CompressFile(queue &q, std::string &input_file,
                 std::vector<std::string> outfilenames, int iterations,
                 bool report) {
  size_t isz;
  char *pinbuf;

  auto dev = q.get_device();
  auto ctxt = q.get_context();
  usm_allocator<GzipOutInfo, usm::alloc::host> alloc_GzipOutInfo(ctxt, dev);
  usm_allocator<unsigned, usm::alloc::host> alloc_unsigned(ctxt, dev);
  usm_allocator<char *, usm::alloc::host> alloc_char_ptr(ctxt, dev);
  usm_allocator<char, usm::alloc::host> alloc_char(ctxt, dev);

  // Read the input file
  std::string device_string =
      q.get_device().get_info<info::device::name>().c_str();

  // If
  // the device is S10, we pre-pin some buffers to
  // improve DMA performance, which is needed to
  // achieve peak kernel throughput. Pre-pinning is
  // only supported on the PAC-S10-USM BSP. It's not
  // needed on PAC-A10 to achieve peak performance.
  bool isS10 = (device_string.find("s10") != std::string::npos);
  bool prepin = q.get_device().has(aspect::usm_host_allocations);

  if (isS10 && !prepin) {
    std::cout << "Warning: Host allocations are not supported on this "
                 "platform, which means that pre-pinning is not supported. DMA "
                 "transfers may be slower than expected which may reduce "
                 "application throughput.\n\n";
  }

  // padding for the input and output buffers to deal with granularity of
  // kernel reads and writes
  constexpr size_t kInOutPadding = 16 * kVec;  
  
  std::ifstream file(input_file,
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    isz = file.tellg();
    if (prepin) {
      pinbuf = (char *)malloc_host(
          isz + kInOutPadding, q.get_context());  // Pre-pin the buffer, for faster DMA
    } else {                      // throughput, using malloc_host().
      pinbuf = new char[isz + kInOutPadding];
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

  // Array of kernel info structures...
  struct KernelInfo *kinfo[kNumEngines];

  for (size_t eng = 0; eng < kNumEngines; eng++) {
    kinfo[eng] = new struct KernelInfo[buffers_count];
    if (kinfo[eng] == NULL) {
      std::cout << "Cannot allocate kernel info buffer.\n";
      return 1;
    }
  }

  // This loop allocates host-side USM buffers, to be accessed by the kernel.
  for (size_t eng = 0; eng < kNumEngines; eng++) {
    for (int i = 0; i < buffers_count; i++) {
      kinfo[eng][i].input_size = isz;
      // Allocating slightly larger buffers (+ 16 * kVec) to account for
      // granularity of kernel writes
      kinfo[eng][i].output_size =
          ((isz + kInOutPadding) < kMinBufferSize) ? kMinBufferSize
                                                   : (isz + kInOutPadding);
      const size_t input_alloc_size = isz + kInOutPadding;

      kinfo[eng][i].last_block = true;
      kinfo[eng][i].pref_buffer = pinbuf;

      // Only allocate N_BUFFERING number of buffers and reuse them on
      // subsequent iterations.
      kinfo[eng][i].gzip_out_buf =
          i >= N_BUFFERING ? kinfo[eng][i - N_BUFFERING].gzip_out_buf
                           : alloc_GzipOutInfo.allocate(BATCH_SIZE *
                                                        sizeof(GzipOutInfo));
      kinfo[eng][i].current_crc =
          i >= N_BUFFERING
              ? kinfo[eng][i - N_BUFFERING].current_crc
              : alloc_unsigned.allocate(BATCH_SIZE * sizeof(uint32_t));

      for (int b = 0; b < BATCH_SIZE; b++) {
        kinfo[eng][i].current_crc[b] = 0;
        kinfo[eng][i].gzip_out_buf[b].compression_sz = 0;
      }

      // Allocate space for the array of pointers. The array contains
      // BATCH_SIZE number of pointers.
      kinfo[eng][i].pibuf_ptr_array =
          i >= N_BUFFERING
              ? kinfo[eng][i - N_BUFFERING].pibuf_ptr_array
              : alloc_char_ptr.allocate(BATCH_SIZE * sizeof(char *));
      kinfo[eng][i].pobuf_ptr_array =
          i >= N_BUFFERING
              ? kinfo[eng][i - N_BUFFERING].pobuf_ptr_array
              : alloc_char_ptr.allocate(BATCH_SIZE * sizeof(char *));

      // For each pointer, allocated space for the input/output buffers
      if (i <
          N_BUFFERING) {  // But only for the first N_BUFFERING kinfo structs
                          // since the buffers get subsequently reused.
        for (int b = 0; b < BATCH_SIZE; b++) {
          kinfo[eng][i].pibuf_ptr_array[b] =
              alloc_char.allocate(input_alloc_size * sizeof(char));
          kinfo[eng][i].pobuf_ptr_array[b] =
              alloc_char.allocate(kinfo[eng][i].output_size * sizeof(char));
          memset(kinfo[eng][i].pobuf_ptr_array[b], 0,
                 kinfo[eng][i].output_size);  // Initialize output buf to zero.
        }
      }
    }
  }

  // Vectors to store the in/out pointers for each iteration (buffers_count),
  // for each engine (kNumengines), for each batch (BATCH_SIZE).
  std::vector<std::array<std::array<char *, BATCH_SIZE>, kNumEngines>> in_ptrs(
      buffers_count);
  std::vector<std::array<std::array<char *, BATCH_SIZE>, kNumEngines>> out_ptrs(
      buffers_count);

  // Grab the pointers and populate the vectors
  for (size_t index = 0; index < buffers_count; index++) {
    for (size_t eng = 0; eng < kNumEngines; eng++) {
      for (size_t i = 0; i < BATCH_SIZE; i++) {
        if (i < BATCH_SIZE) {
          in_ptrs[index][eng][i] = kinfo[eng][index].pibuf_ptr_array[i];
          out_ptrs[index][eng][i] = kinfo[eng][index].pobuf_ptr_array[i];
        } else {  // Re-use first pointer to avoid invalid-arg runtime error
          in_ptrs[index][eng][i] = kinfo[eng][index].pibuf_ptr_array[0];
          out_ptrs[index][eng][i] = kinfo[eng][index].pobuf_ptr_array[0];
        }
      }
    }
  }

  /*************************************************/
  /* Main loop where the actual execution happens  */
  /*************************************************/

  // Initialize the input buffer with the file to be compressed.
  // In this reference design, for simplicity, we use the same input file
  // repeatedly. We also do not bother to copy the output (the compressed
  // result) out of the output buffers since it's the same on every execution.
  // Recall that the input and output buffers are reused, therefore in a real
  // application you'll need to both copy new files into the input buffers and
  // copy the compressed results out of the output buffers. System throughput
  // may degrade if these host-side operations take longer than the kernel
  // execution time because you won't be able to safely invoke a new kernel
  // until the copies are complete, leading to deadtime between kernel
  // executions. The host-side processing time can be hidden using "N-way
  // buffering" -- see the corresponding tutorial (n_way_buffering) to learn how
  // this is can be done.
  int initial_index = std::min(N_BUFFERING, buffers_count);
  for (int index = 0; index < initial_index; index++) {
    for (size_t eng = 0; eng < kNumEngines; eng++) {
      for (int b = 0; b < BATCH_SIZE; b++) {
        memcpy(kinfo[eng][index].pibuf_ptr_array[b],
               kinfo[eng][index].pref_buffer, kinfo[eng][index].input_size);
      }
    }
  }

#ifdef FPGA_EMULATOR
#elif FPGA_SIMULATOR
#else
  auto start = std::chrono::steady_clock::now();
#endif

  // Launch initial set of kernels
  for (int index = 0; index < initial_index; index++) {
    for (size_t eng = 0; eng < kNumEngines; eng++) {
      kinfo[eng][index].kernel_event = SubmitGzipTasks(q,
                                                    kinfo[eng][index].input_size,
                                                    kinfo[eng][index].gzip_out_buf,
                                                    kinfo[eng][index].current_crc, 
                                                    kinfo[eng][index].last_block,
                                                    {},
                                                    in_ptrs[index][eng],
                                                    out_ptrs[index][eng],
                                                    eng
                                                    );
    }
  }

  // Main loop where the gzip engine is repeatedly invoked in a double-buffered
  // fashion.
  for (int index = initial_index; index < buffers_count; index++) {
    for (size_t eng = 0; eng < kNumEngines; eng++) {
      /************************************/
      /************************************/
      /*         LAUNCH GZIP ENGINE       */
      /************************************/
      /************************************/

      kinfo[eng][index].kernel_event = SubmitGzipTasks(q,
                                                  kinfo[eng][index].input_size, 
                                                  kinfo[eng][index].gzip_out_buf,
                                                  kinfo[eng][index].current_crc,
                                                  kinfo[eng][index].last_block,
                                                  kinfo[eng][index - N_BUFFERING].kernel_event,
                                                  in_ptrs[index][eng],
                                                  out_ptrs[index][eng],                                                  
                                                  eng
                                                  );

    }
  }

  // Wait for all kernels to complete.
  for (int index = buffers_count - initial_index; index < buffers_count;
       index++) {
    for (size_t eng = 0; eng < kNumEngines; eng++) {
      for (auto event : kinfo[eng][index].kernel_event) {
        event.wait();
      }
    }
  }

// Stop the timer.
#ifdef FPGA_EMULATOR
#elif FPGA_SIMULATOR
#else
  auto end = std::chrono::steady_clock::now();
  double diff_total = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
  if (report) {
    std::cout << "Total execution time: " << (double)diff_total * 1000000
              << "us \n";
    std::cout << "Average per batch_latency: "
              << (double)diff_total * 1000000 / iterations << " us \n";
  }
  double gbps = BATCH_SIZE * iterations * isz / (double)diff_total /
                1000000000.0;
#endif

  // Sanity check the compressed size of every result. Also sum the compressed
  // sizes together to be later used to calculate the compression ratio.
  size_t compressed_sz[kNumEngines];
  for (int eng = 0; eng < kNumEngines; eng++) {
    compressed_sz[eng] = 0;
    for (int index = 0; index < buffers_count; index++) {
      for (int b = 0; b < BATCH_SIZE; b++) {
        if (kinfo[eng][index].gzip_out_buf[b].compression_sz >
            kinfo[eng][index].input_size) {
          std::cerr << "Unsupported: compressed file larger than input file ( "
                    << kinfo[eng][index].gzip_out_buf[b].compression_sz
                    << " bytes)\n";
          return 1;
        }
        compressed_sz[eng] += kinfo[eng][index].gzip_out_buf[b].compression_sz;
      }
    }
  }

  if (report) std::cout << "Writing gzip archive to disk and verifying\n";
  // Write the outputs from buffer set 0 and check for errors
  for (int i = 0; i < 1;
       i++) {  // Here you could iterate through all the buffer sets.
    for (int eng = 0; eng < kNumEngines; eng++) {
      for (int b = 0; b < BATCH_SIZE; b++) {
        if (report &&
            WriteBlockGzip(
                input_file, outfilenames[eng], kinfo[eng][i].pobuf_ptr_array[b],
                kinfo[eng][i].gzip_out_buf[b].compression_sz,
                kinfo[eng][i].input_size,
                Crc32(kinfo[eng][i].pref_buffer, kinfo[eng][i].input_size,
                      kinfo[eng][i].current_crc[b])  // Compute the remaining
                                                     // piece of the CRC.
                )) {
          std::cout << "FAILED\n";
          return 1;
        }
      }
    }
  }

  // Decompress the output from engine-0 and compare against the input file.
  // Only engine-0's output is verified since all engines are fed the same input
  // data.
  if (report && CompareGzipFiles(input_file, outfilenames[0])) {
    std::cout << "FAILED\n";
    return 1;
  }

  // Generate throughput report
  // First gather all the execution times.
  size_t time_k_crc[kNumEngines];
  size_t time_k_lz[kNumEngines];
  size_t time_k_huff[kNumEngines];
  
  for (int eng = 0; eng < kNumEngines; eng++) {
    time_k_crc[eng] = 0;
    time_k_lz[eng] = 0;
    time_k_huff[eng] = 0;
    for (int i = 0; i < buffers_count;
         i++) {  // Execution times (total sums) of the individual gzip kernels
      time_k_crc[eng] +=
          SyclGetExecTimeNs(kinfo[eng][i].kernel_event[kCRCIndex]);
      time_k_lz[eng] +=
          SyclGetExecTimeNs(kinfo[eng][i].kernel_event[kLZReductionIndex]);
      time_k_huff[eng] +=
          SyclGetExecTimeNs(kinfo[eng][i].kernel_event[kStaticHuffmanIndex]);
    }
  }

  if (report) {
    double compression_ratio = (double)((double)compressed_sz[0] / (double)isz /
                                        BATCH_SIZE / iterations);
#ifdef FPGA_EMULATOR
#elif FPGA_SIMULATOR
#else
    std::cout << "Throughput: " << kNumEngines * gbps << " GB/s\n\n";
    for (int eng = 0; eng < kNumEngines; eng++) {
      std::cout << "TP breakdown for engine #" << eng << " (GB/s)\n";
      std::cout << "CRC = " << BATCH_SIZE * iterations * isz / (double)time_k_crc[eng]
                << "\n";
      std::cout << "LZ77 = " << BATCH_SIZE * iterations * isz / (double)time_k_lz[eng]
                << "\n";
      std::cout << "Huffman Encoding = "
                << BATCH_SIZE * iterations * isz / (double)time_k_huff[eng] << "\n";
    }
#endif
    std::cout << "Compression Ratio " << compression_ratio * 100 << "%\n";
  }

  // Cleanup anything that was allocated by this routine.
  // delete the file mapping now that all kernels are complete, and we've
  // snapped the time delta

  // delete the file mapping now that all kernels are complete, and we've
  // snapped the time delta
  if (prepin) {
    free(pinbuf, q.get_context());
  } else {
    delete pinbuf;
  }

  for (int eng = 0; eng < kNumEngines; eng++) {
    for (int i = 0; i < buffers_count; i++) {
      if (i < N_BUFFERING) {
        alloc_GzipOutInfo.deallocate(kinfo[eng][i].gzip_out_buf,
                                     BATCH_SIZE * sizeof(GzipOutInfo));
        alloc_unsigned.deallocate(kinfo[eng][i].current_crc,
                                  BATCH_SIZE * sizeof(uint32_t));

        // dealloc the input buffers
        for (int b = 0; b < BATCH_SIZE; b++) {
          alloc_char.deallocate(kinfo[eng][i].pibuf_ptr_array[b],
                                kinfo[eng][i].input_size * sizeof(char));
          alloc_char.deallocate(kinfo[eng][i].pobuf_ptr_array[b],
                                kinfo[eng][i].output_size * sizeof(char));
        }

        // dealloc the array of input buffer pointers
        alloc_char_ptr.deallocate(kinfo[eng][i].pibuf_ptr_array,
                                  BATCH_SIZE * sizeof(char *));
        alloc_char_ptr.deallocate(kinfo[eng][i].pobuf_ptr_array,
                                  BATCH_SIZE * sizeof(char *));
      }
    }
    delete[] kinfo[eng];
  }

  if (report) std::cout << "PASSED\n";
  return 0;
}
