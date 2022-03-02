#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <thread>
#include <utility>
#include <vector>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

#include "common.hpp"
#include "constexpr_math.hpp"
#include "gzip_decompressor.hpp"
#include "gzip_header_data.hpp"
#include "simple_crc32.hpp"

using namespace sycl;
using namespace std::chrono;

// the number of literals to process per cycle can be set from the command line
// use the macro -DLITERALS_PER_CYCLE=<literals_per_cycle>
// This is sent to the LZ77 decoder to read multiple elements per cycle from
// the history buffer.
#ifndef LITERALS_PER_CYCLE
#define LITERALS_PER_CYCLE 4
#endif
constexpr unsigned kLiteralsPerCycle = LITERALS_PER_CYCLE;
static_assert(fpga_tools::IsPow2(kLiteralsPerCycle));

// declare the kernel and pipe names globally to reduce name mangling
class ProducerID;
class ConsumerID;
class InPipeID;
class OutPipeID;

using InPipe = ext::intel::pipe<InPipeID, unsigned char>;
using OutPipe =
    ext::intel::pipe<OutPipeID, FlagBundle<BytePack<kLiteralsPerCycle>>>;

////////////////////////////////////////////////////////////////////////////////
bool DecompressFile(queue&, std::string, std::string, int, bool, bool);
bool DecompressTest(queue&, std::string);
unsigned GetGZIPUncompressedSize(std::vector<unsigned char>);
void PrintUsage(std::string);
event SubmitProducer(queue&, int, unsigned char*);
event SubmitConsumer(queue&, int, unsigned char*, int*);
std::vector<unsigned char> ReadInputFile(std::string filename);
void WriteOutputFile(std::string, std::vector<unsigned char>&);
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  // reading and validating the command line arguments
  // if no arguments are given, we will run the default tests for uncompressed,
  // statically compressed, and dynamically compressed blocks
  // if arguments are given, we will assume the user wants to decompress a 
  // specific file
  std::string test_dir = "../data";
  std::string in_filename;
  std::string out_filename;
  int runs;
  bool default_test_mode = false;
  if (argc == 1 || argc == 2) {
    default_test_mode = true;
  } else if (argc > 4) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (default_test_mode) {
    if (argc > 1) test_dir = argv[1];
  } else {
    // default the number of runs based on emulation, simulation, or hardware
#if defined(FPGA_EMULATOR)
    runs = 2;
#elif defined(FPGA_SIMULATOR)
    runs = 1;
#else
    runs = 9;
#endif

    in_filename = argv[1];
    out_filename = argv[2];
    if (argc > 3) runs = atoi(argv[3]);
    if (runs < 1) {
      std::cerr << "ERROR: 'runs' must be greater than 0\n";
      std::terminate();
    }
  }

  // the device selector
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector selector;
#elif defined(FPGA_SIMULATOR)
  std::string simulator_device_string =
      "SimulatorDevice : Multi-process Simulator (aclmsim0)";
  select_by_string selector = select_by_string{simulator_device_string};
#else
  sycl::ext::intel::fpga_selector selector;
#endif

  // create the device queue
  queue q(selector, dpc_common::exception_handler);

  bool passed;
  if (default_test_mode) {
    passed = DecompressTest(q, test_dir);
  } else {
    passed = DecompressFile(q, in_filename, out_filename, runs, true, true);
  }

  if (passed) {
    std::cout << "PASSED" << std::endl;
    return 0;
  } else {
    std::cout << "FAILED" << std::endl;
    return 1;
  }
}

//
// The default decompression test. This is called if the user did not provide
// any command line arguments and tests decompressing uncompressed, statically-,
// and dynamically-compressed blocks
//
bool DecompressTest(queue& q, std::string test_dir) {
  std::string uncompressed_filename = test_dir + "/uncompressed.gz";
  std::string static_compress_filename = test_dir + "/static_compressed.gz";
  std::string dynamic_compress_filename = test_dir + "/dynamic_compressed.gz";
  std::string tp_test_filename = test_dir + "/tp_test.gz";

  auto print_test_result = [](std::string test_name, bool passed) {
    if (passed)
      std::cout << ">>>>> " << test_name << ": PASSED <<<<<\n";
    else
      std::cerr << ">>>>> " << test_name << ": FAILED <<<<<\n";
  };
  
  std::cout << ">>>>> Uncompressed File Test <<<<<" << std::endl;
  bool uncompressed_test_pass =
      DecompressFile(q, uncompressed_filename, "", 1, false, false);
  print_test_result("Uncompressed File Test", uncompressed_test_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Statically Compressed File Test <<<<<" << std::endl;
  bool static_test_pass =
      DecompressFile(q, static_compress_filename, "", 1, false, false);
  print_test_result("Statically Compressed File Test", static_test_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Dynamically Compressed File Test <<<<<" << std::endl;
  bool dynamic_test_pass =
      DecompressFile(q, dynamic_compress_filename, "", 1, false, false);
  print_test_result("Dynamically Compressed File Test", dynamic_test_pass);
  std::cout << std::endl;

  std::cout << ">>>>> Throughput Test <<<<<" << std::endl;
  constexpr int kTPTestRuns = 5;
  bool tp_test_pass =
      DecompressFile(q, tp_test_filename, "", kTPTestRuns, true, false);
  print_test_result("Throughput Test", tp_test_pass);
  std::cout << std::endl;

  return uncompressed_test_pass && static_test_pass && dynamic_test_pass && tp_test_pass;
}

//
// Decompress a specific file (f_in) and store the contents in f_out
//
bool DecompressFile(queue& q, std::string f_in, std::string f_out, int runs,
                    bool print_stats, bool write_output) {
  // read bytes from the input file
  std::vector<unsigned char> in_bytes = ReadInputFile(f_in);
  int in_count = in_bytes.size();

  // read the expected output size from the last 4 bytes of the file
  // this will let us intelligently size the output buffer
  unsigned out_count = GetGZIPUncompressedSize(in_bytes);
  std::vector<unsigned char> out_bytes(out_count);

  // round up the output count to the nearest multiple of kLiteralsPerCycle,
  // which allows us to not predicate the last writes to the output buffer from
  // the device.
  int out_count_padded =
      fpga_tools::RoundUpToMultiple(out_count, kLiteralsPerCycle);

  // host variables for output from the device
  // the number of bytes read in the Consumer kernel, which should match
  // the uncompressed size read in the footer of the GZIP file (count_h below)
  int decompressed_count_h = 0; 
  
  // the GZIP header data. This is parsed by the GZIPMetadataReader kernel
  GzipHeaderData hdr_data_h;

  // the GZIP footer data. This is parsed by the GZIPMetadataReader kernel.
  // count_h should match decompressed_count_h (we check that later).
  unsigned int crc_h, count_h;

  // track timing information in ms
  std::vector<double> time_ms(runs);

  // input and output data pointers on the device using USM device allocations
  // the input bytes (the bytes of the GZIP file)
  unsigned char *in;

  // the output bytes (the result of the decompression)
  unsigned char *out;

  // the number of decompressed bytes (the number of bytes in 'out')
  // returned by Consumer kernel
  int *decompressed_count;

  // the GZIP header data (see gzip_header_data.hpp)
  GzipHeaderData *hdr_data;

  // the GZIP footer data, where 'count' is the expected number of bytes
  // in the uncompressed file ('out')
  int *crc, *count;

  bool passed = true;

  try {
    // allocate memory on the device
    if ((in = malloc_device<unsigned char>(in_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out = malloc_device<unsigned char>(out_count_padded, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out'\n";
      std::terminate();
    }
    if ((decompressed_count = malloc_device<int>(1, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'decompressed_count'\n";
      std::terminate();
    }
    if ((hdr_data = malloc_device<GzipHeaderData>(1, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'hdr_data'\n";
      std::terminate();
    }
    if ((crc = malloc_device<int>(1, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'crc'\n";
      std::terminate();
    }
    if ((count = malloc_device<int>(1, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'count'\n";
      std::terminate();
    }

    // copy the input data to the device memory and wait for the copy to finish
    q.memcpy(in, in_bytes.data(), in_count * sizeof(unsigned char)).wait();

    std::cout << "Decompressing '" << f_in << "' " << runs
              << ((runs == 1) ? " time" : " times") << std::endl;

    // run the design multiple times to increase the accuracy of the timing
    for (int i = 0; i < runs; i++) {
      std::cout << "Launching kernels for run " << i << std::endl;

      // start the producer and consumer kernels
      auto producer_event = SubmitProducer(q, in_count, in);
      auto consumer_event =
          SubmitConsumer(q, out_count_padded, out, decompressed_count);

      // start the decompression kernels
      auto gzip_decompress_events = SubmitGzipDecompressKernels<InPipe, OutPipe, kLiteralsPerCycle>(q, in_count, hdr_data, crc, count);

      // wait for the producer and consumer to finish
      auto start = high_resolution_clock::now();
      producer_event.wait();
      consumer_event.wait();
      auto end = high_resolution_clock::now();

      // wait for the decompression kernels to finish
      for (auto& e : gzip_decompress_events) { e.wait(); }

      std::cout << "All kernels have finished for run " << i << std::endl;

      // calculate the time the kernels ran for, in milliseconds
      time_ms[i] = duration<double, std::milli>(end - start).count();

      // Copy the output back from the device
      q.memcpy(out_bytes.data(), out, out_count * sizeof(unsigned char)).wait();
      q.memcpy(&decompressed_count_h, decompressed_count, sizeof(int)).wait();
      q.memcpy(&hdr_data_h, hdr_data, sizeof(GzipHeaderData)).wait();
      q.memcpy(&crc_h, crc, sizeof(int)).wait();
      q.memcpy(&count_h, count, sizeof(int)).wait();

      // validating the output
      // check the magic header we read
      if (hdr_data_h.MagicNumber() != 0x1f8b) {
        auto save_flags = std::cerr.flags();
        std::cerr << "ERROR: Incorrect magic header value of 0x"
                  << std::hex << std::setw(4) << std::setfill('0')
                  << hdr_data_h.MagicNumber() << " (should be 0x1f8b)\n";
        std::cerr.flags(save_flags);
        passed = false;
      }

      // check the number of bytes we read
      if (count_h != out_count) {
        std::cerr << "ERROR: Out counts do not match: "
                  << count_h << " != " << out_count
                  << "(count_h != out_count)\n";
        passed = false;
      }

      // validate the decompressed number of bytes based on the expectation
      if (decompressed_count_h != count_h) {
        std::cerr << "ERROR: decompressed_count_h != count_h ("
                  << decompressed_count_h << " != " << count_h << ")\n";
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
  } catch (exception const& e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::terminate();
  }

  // free the allocated device memory
  sycl::free(in, q);
  sycl::free(out, q);
  sycl::free(decompressed_count, q);
  sycl::free(hdr_data, q);
  sycl::free(crc, q);
  sycl::free(count, q);

  // write the output file
  if (write_output) {
    std::cout << std::endl;
    std::cout << "Writing output data to '" << f_out << "'" << std::endl;
    std::cout << std::endl;
    WriteOutputFile(f_out, out_bytes);
  }

  // print the performance results
  if (passed && print_stats) {
    // NOTE: when run in emulation, these results do not accurately represent
    // the performance of the kernels on real FPGA hardware
    double avg_time_ms;
    if (runs > 1) {
      avg_time_ms =
          std::accumulate(time_ms.begin() + 1, time_ms.end(), 0.0) / (runs - 1);
    } else {
      avg_time_ms = time_ms[0];
    }

    double compression_ratio
        = (double)(decompressed_count_h) / (double)(in_count);

    // the number of input and output megabytes, respectively
    size_t out_mb = decompressed_count_h * sizeof(unsigned char) * 1e-6;

    std::cout << "Execution time: " << avg_time_ms << " ms\n";
    std::cout << "Output Throughput: " << (out_mb / (avg_time_ms * 1e-3))
              << " MB/s\n";
    std::cout << "Compression Ratio: " << compression_ratio << "\n";
  }

  return passed;
}

//
// Produce bytes from in_ptr to InPipe
//
event SubmitProducer(queue& q, int count, unsigned char* in_ptr) {
  return q.single_task<ProducerID>([=] {
    device_ptr<unsigned char> in(in_ptr);
    for (int i = 0; i < count; i++) {
      InPipe::write(in[i]);
    }
  });
}

//
// Consume bytes from OutPipe and write them to in_ptr
//
event SubmitConsumer(queue& q, int out_count_padded, unsigned char* out_ptr, int* inflated_count_ptr) {
  const int out_iterations = (out_count_padded / kLiteralsPerCycle);
  return q.submit([&](handler &h) {
    h.single_task<ConsumerID>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<unsigned char> out(out_ptr);
      device_ptr<int> decompressed_count(inflated_count_ptr);

      int i = 0;
      bool i_in_range = 0 < out_iterations;
      bool i_next_in_range = 1 < out_iterations;
      int valid_byte_count = 0;
      bool done;

      do {
        // read the pipe data
        bool valid_pipe_read;
        auto pipe_data = OutPipe::read(valid_pipe_read);
        done = pipe_data.flag && valid_pipe_read;

        if (!done && valid_pipe_read) {
          // guard against overrunning the output array.
          if (i_in_range) {
            #pragma unroll
            for (int j = 0; j < kLiteralsPerCycle; j++) {
              out[i * kLiteralsPerCycle + j] = pipe_data.data.byte[j];
            }
          }
          valid_byte_count += pipe_data.data.valid_count;
          i_in_range = i_next_in_range;
          i_next_in_range = i < out_iterations - 2;
          i++;
        }
      } while (!done);

      *decompressed_count = valid_byte_count;
    });
  });
}

//
// Reads 'filename' and returns an array of chars that are the bytes of the file
//
std::vector<unsigned char> ReadInputFile(std::string filename) {
  // open file stream
  std::ifstream fin(filename);

  // make sure it opened
  if (!fin.good() || !fin.is_open()) {
    std::cerr << "ERROR: could not open " << filename << " for reading\n";
    std::terminate();
  }

  // read in bytes
  std::vector<unsigned char> result;
  char tmp;
  while (fin.get(tmp)) { result.push_back(tmp); }
  fin.close();
  
  return result;
}

//
// Writes the chars (bytes) from 'data' to 'filename'
//
void WriteOutputFile(std::string filename, std::vector<unsigned char>& data) {
  // open file stream
  std::ofstream fout(filename.c_str());

  // make sure it opened
  if (!fout.good() || !fout.is_open()) {
    std::cerr << "ERROR: could not open " << filename << " for writing\n";
    std::terminate();
  }

  // write out bytes
  for (auto& c : data) { fout << c; }
  fout.close();
}

//
// Gets the uncompressed size from the bytes of a GZIP file
//
unsigned GetGZIPUncompressedSize(std::vector<unsigned char> bytes) {
  std::vector<unsigned char> last_4_bytes(bytes.end() - 4, bytes.end());
  return *(reinterpret_cast<unsigned*>(last_4_bytes.data()));
}

//
// Prints the usage for main
//
void PrintUsage(std::string exe_name) {
  std::cerr << "USAGE: \n"
            << exe_name << " <input filename> <output filename> [runs]\n"
            << exe_name << " <test directory>" << std::endl;
}
