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
#include "gzip_decompressor.hpp"
#include "gzip_header_data.hpp"
#include "mp_math.hpp"
#include "simple_crc32.hpp"

using namespace sycl;
using namespace std::chrono;

// the number of literals to process per cycle can be set from the command line
// use the macro -DLITERALS_PER_CYCLE=<literals_per_cycle>
#ifndef LITERALS_PER_CYCLE
#define LITERALS_PER_CYCLE 2
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
    ext::intel::pipe<OutPipeID, FlagBundle<LiteralPack<kLiteralsPerCycle>>>;

////////////////////////////////////////////////////////////////////////////////
void PrintUsage(std::string);
event SubmitProducer(queue&, int, unsigned char*);
event SubmitConsumer(queue&, int, unsigned char*, int*);
std::vector<unsigned char> ReadInputFile(std::string filename);
void WriteOutputFile(std::string, std::vector<unsigned char>&);
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  // make sure we have an acceptable number of arguments
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  bool passed = true;

  // reading and validating the command line arguments
  std::string in_filename;
  std::string out_filename;
  
#ifdef FPGA_EMULATOR
  int runs = 2;
#elif FPGA_SIMULATOR
  int runs = 2;
#else
  int runs = 8;
#endif

  // get input and output filenames
  if (argc > 1) {
    in_filename = argv[1];
  }
  if (argc > 2) {
    out_filename = argv[2];
  }
  if (argc > 3) {
    runs = atoi(argv[3]);
  }

  // enforce at least two runs
  if (runs < 2) {
    std::cerr << "ERROR: 'runs' must be 2 or more\n";
    std::terminate();
  }

  // the device selector
#ifdef FPGA_EMULATOR
  sycl::ext::intel::fpga_emulator_selector selector;
#elif FPGA_SIMULATOR
  std::string simulator_device_string =
      "SimulatorDevice : Multi-process Simulator (aclmsim0)";
  select_by_string selector = select_by_string{simulator_device_string};
#else
  sycl::ext::intel::fpga_selector selector;
#endif

  // create the device queue
  queue q(selector, dpc_common::exception_handler);

  // read input file
  std::vector<unsigned char> in_bytes = ReadInputFile(in_filename);
  int in_count = in_bytes.size();

  // read the expected output size from the last 4 bytes of the file
  std::vector<unsigned char> last_4_bytes(in_bytes.end() - 4, in_bytes.end());
  unsigned out_count = *(reinterpret_cast<unsigned*>(last_4_bytes.data()));
  std::vector<unsigned char> out_bytes(out_count);

  // round up the output count to the nearest multiple of kLiteralsPerCycle
  // this allows to ignore predicating the last writes to the output
  int out_count_padded =
      fpga_tools::RoundUpToMultiple(out_count, kLiteralsPerCycle);

  // host variables for output from device
  int decompressed_count_h = 0;
  GzipHeaderData hdr_data_h;
  unsigned int crc_h, count_h;

  // track timing information in ms
  std::vector<double> time(runs);

  // input and output data pointers on the device using USM device allocations
  unsigned char *in, *out;
  int *decompressed_count;
  GzipHeaderData *hdr_data;
  int *crc;
  int *count;

  try {
    // allocate memory on the device for the input and output
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

    std::cout << "Decompressing '" << in_filename << "' " << runs << " times"
              << std::endl;
    // run the design multiple times to increase the accuracy of the timing
    for (int i = 0; i < runs; i++) {
      std::cout << "Launching kernels for run " << i << std::endl;

      // run the producer and consumer kernels
      auto producer_event = SubmitProducer(q, in_count, in);
      auto consumer_event =
          SubmitConsumer(q, out_count_padded, out, decompressed_count);

      // run the decompression kernels
      auto gzip_decompress_events = SubmitGzipDecompressKernels<InPipe, OutPipe, kLiteralsPerCycle>(q, in_count, hdr_data, crc, count);

      // wait for the producer and consumer to finish
      auto start = high_resolution_clock::now();
      producer_event.wait();
      consumer_event.wait();
      auto end = high_resolution_clock::now();

      // wait for the decompression kernels to finish
      for (auto& e : gzip_decompress_events) { e.wait(); }

      std::cout << "All kernels finished for run " << i << std::endl;

      // calculate the time the kernels ran for, in milliseconds
      time[i] = duration<double, std::milli>(end - start).count();

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
      // keep the first decompressed_count_h bytes
      if (decompressed_count_h != count_h) {
        std::cerr << "ERROR: decompressed_count_h != count_h ("
                  << decompressed_count_h << " != " << count_h << ")\n";
        passed = false;
      }

      // compute and check the CRC of the output data
      auto crc32_exp = SimpleCRC32(0, out_bytes.data(), out_count);
      if (crc_h != crc32_exp) {
        auto save_flags = std::cout.flags();
        std::cerr << std::hex << std::setw(4) << std::setfill('0');
        std::cerr << "ERROR: output data CRC does not match the expected CRC "
                  << "0x" << crc_h << " != 0x" << crc32_exp
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

  // write output file
  std::cout << std::endl;
  std::cout << "Writing output data to '" << out_filename << "'" << std::endl;
  std::cout << std::endl;
  WriteOutputFile(out_filename, out_bytes);

  // print the performance results
  if (passed) {
    // NOTE: when run in emulation, these results do not accurately represent
    // the performance of the kernels on real FPGA hardware
    double avg_time_ms =
        std::accumulate(time.begin() + 1, time.end(), 0.0) / (runs - 1);

    double compression_ratio
        = (double)(decompressed_count_h) / (double)(in_count);

    // the number of input and output megabytes, respectively
    size_t in_mb = in_count * sizeof(unsigned char) * 1e-6;
    size_t out_mb = decompressed_count_h * sizeof(unsigned char) * 1e-6;

    std::cout << "Execution time: " << avg_time_ms << " ms\n";
    std::cout << "Output Throughput: " << (out_mb / (avg_time_ms * 1e-3))
              << " MB/s\n";
    std::cout << "Compression Ratio: " << compression_ratio << "\n";
    std::cout << "Input Throughput: " << (in_mb / (avg_time_ms * 1e-3))
              << " MB/s\n";
    std::cout << std::endl;

    std::cout << "PASSED" << std::endl;
    return 0;
  } else {
    std::cout << "FAILED" << std::endl;
    return 1;
  }
}

//
// Produce bytes from in_ptr to InPipe
//
event SubmitProducer(queue& q, int count, unsigned char* in_ptr) {
  return q.single_task<ProducerID>([=] {
    device_ptr<unsigned char> in(in_ptr);
    for (int i = 0; i < count; i++) {
      unsigned char d = in[i];
      InPipe::write(d);
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
              out[i * kLiteralsPerCycle + j] = pipe_data.data.literal[j];
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
  std::ifstream fin(filename);
  if (!fin.good() || !fin.is_open()) {
    std::cerr << "ERROR: could not open " << filename << " for reading\n";
    std::terminate();
  }
  std::vector<unsigned char> result;
  char tmp;
  while (fin.get(tmp)) {
    result.push_back(tmp);
  }
  fin.close();
  
  return result;
}

//
// Writes the chars (bytes) from 'data' to 'filename'
//
void WriteOutputFile(std::string filename, std::vector<unsigned char>& data) {
  std::ofstream fout(filename.c_str());
  if (!fout.good() || !fout.is_open()) {
    std::cerr << "ERROR: could not open " << filename << " for writing\n";
    std::terminate();
  }

  for (auto& c : data) {
    fout << c;
  }
  fout.close();
}

//
// Prints the usage for main
//
void PrintUsage(std::string exe_name) {
  std::cerr << "USAGE: " << exe_name
            << " <input filename> <output filename> [runs]\n";
}
