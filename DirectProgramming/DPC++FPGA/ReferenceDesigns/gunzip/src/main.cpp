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

#include "ByteHistory.hpp"
#include "Decompressor.hpp"
#include "HeaderData.hpp"
#include "mp_math.hpp"

using namespace sycl;
using namespace std::chrono;

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

class ProducerID;
class ConsumerID;
class InPipeID;
class OutPipeID;

#ifndef LITERALS_PER_CYCLE
#define LITERALS_PER_CYCLE 4
#endif
constexpr unsigned LiteralsPerCycle = LITERALS_PER_CYCLE;
static_assert(fpga_tools::IsPow2(LiteralsPerCycle));

using InPipe = ext::intel::pipe<InPipeID, char>;
using OutPipe = ext::intel::pipe<OutPipeID, FlagBundle<LiteralPack<LiteralsPerCycle>>>;

////////////////////////////////////////////////////////////////////////////////
event SubmitProducer(queue&, unsigned char*, int);
event SubmitConsumer(queue&, unsigned char*, int*);
std::vector<unsigned char> ReadInputFile(std::string filename);
void WriteOutputFile(std::string, std::vector<unsigned char>&);
////////////////////////////////////////////////////////////////////////////////

class select_by_string : public sycl::default_selector {
public:
  select_by_string(std::string s) : target_name(s) {}
  virtual int operator()(const sycl::device &device) const {
    std::string name = device.get_info<sycl::info::device::name>();
    if (name.find(target_name) != std::string::npos) {
      // The returned value represents a priority, this number is chosen to be
      // large to ensure high priority
      return 10000;
    }
    return -1;
  }
 
private:
  std::string target_name;
};

int main(int argc, char* argv[]) {
  bool passed = true;

  // reading and validating the command line arguments
  std::string in_filename = "../data/in.gz";
  std::string out_filename = "../data/out";
  
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

  // round up the output count to the nearest multiple of LiteralsPerCycle
  // this allows to ignore predicating the last writes to the output
  int out_count_padded =
      fpga_tools::RoundUpToMultiple(out_count, LiteralsPerCycle);

  // host variables for output from device
  int inflated_count_host = 0;
  HeaderData hdr_data_host;
  unsigned int crc_host, count_host;

  // track timing information in ms
  std::vector<double> time(runs);

  // input and output data pointers on the device using USM device allocations
  unsigned char *in, *out;
  int *inflated_count ;
  HeaderData *hdr_data;
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
    if ((inflated_count = malloc_device<int>(1, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'inflated_count'\n";
      std::terminate();
    }
    if ((hdr_data = malloc_device<HeaderData>(1, q)) == nullptr) {
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

    // run the design multiple times to increase the accuracy of the timing
    for (int i = 0; i < runs; i++) {
      // run the producer and consumer kernels
      std::cout << "Launching Producer and Consumer kernels" << std::endl;
      auto producer_event = SubmitProducer(q, in, in_count);
      auto consumer_event = SubmitConsumer(q, out, inflated_count);

      // run the decompression kernels
      std::cout << "Launching gzip kernels\n";
      auto decompress_events = SubmitDecompressKernels<InPipe, OutPipe, LiteralsPerCycle>(q, in_count, hdr_data, crc, count);

      // wait for the producer and consumer to finish
      std::cout << "Waiting on producer and consumer kernels" << std::endl;
      auto start = high_resolution_clock::now();
      producer_event.wait();
      consumer_event.wait();
      auto end = high_resolution_clock::now();

      // wait for the decompression kernels to finish
      std::cout << "Waiting on decompress kernels" << std::endl;
      for (auto& e : decompress_events) {
        e.wait();
      }

      std::cout << "Done waiting" << std::endl;

      // calculate the time the kernels ran for, in milliseconds
      time[i] = duration<double, std::milli>(end - start).count();

      // Copy the output back from the device
      q.memcpy(out_bytes.data(), out, out_count * sizeof(unsigned char)).wait();
      q.memcpy(&inflated_count_host, inflated_count, sizeof(int)).wait();
      q.memcpy(&hdr_data_host, hdr_data, sizeof(HeaderData)).wait();
      q.memcpy(&crc_host, crc, sizeof(int)).wait();
      q.memcpy(&count_host, count, sizeof(int)).wait();

      // validate the results
      // keep the first inflated_count_host bytes
      out_bytes.resize(inflated_count_host);
      if (inflated_count_host != count_host) {
        std::cerr << "ERROR: inflated_count_host != count_host ("
                  << inflated_count_host << " != " << count_host << ")\n";
        passed = false;
      }

      // TODO
      std::cout << "inflated_count_host = " << inflated_count_host << " bytes\n";
      std::cout << hdr_data_host;
      std::cout << "crc = " << crc_host << "\n";
      std::cout << "count = " << count_host << "\n";
      std::cout << "\n";
    }
  } catch (exception const& e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::terminate();
  }

  // free the allocated device memory
  sycl::free(in, q);
  sycl::free(out, q);

  // write output file
  std::cout << "Writing output file\n";
  WriteOutputFile(out_filename, out_bytes);

  // print the performance results
  if (passed) {
    // NOTE: when run in emulation, these results do not accurately represent
    // the performance of the kernels in actual FPGA hardware
    double avg_time_ms =
        std::accumulate(time.begin() + 1, time.end(), 0.0) / (runs - 1);

    // TODO: what should the count be here? Input size? Output size?
    //size_t megabytes = in_count * sizeof(unsigned char) * 1e-6;
    size_t megabytes = inflated_count_host * sizeof(unsigned char) * 1e-6;

    std::cout << "Execution time: " << avg_time_ms << " ms\n";
    std::cout << "Throughput: " << (megabytes / (avg_time_ms * 1e-3))
              << " MB/s\n";

    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

//
// TODO
//
event SubmitProducer(queue& q, unsigned char* in_ptr, int count) {
  return q.single_task<ProducerID>([=] {
    device_ptr<unsigned char> in(in_ptr);
    for (int i = 0; i < count; i++) {
      unsigned char d = in[i];
      InPipe::write(d);
    }
  });
}

//
// TODO
//
event SubmitConsumer(queue& q, unsigned char* out_ptr, int* inflated_count_ptr) {
  return q.submit([&](handler &h) {
    h.single_task<ConsumerID>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<unsigned char> out(out_ptr);
      device_ptr<int> inflated_count(inflated_count_ptr);

      int i = 0;
      int valid_byte_count = 0;
      bool done;

      do {
        // read the pipe data
        bool valid_pipe_read;
        auto pipe_data = OutPipe::read(valid_pipe_read);
        done = pipe_data.flag && valid_pipe_read;

        if (!done && valid_pipe_read) {
          #pragma unroll
          for (int j = 0; j < LiteralsPerCycle; j++) {
            out[i * LiteralsPerCycle + j] = pipe_data.data.literal[j];
          }
          valid_byte_count += pipe_data.data.valid_count;
          i++;
        }
      } while (!done);

      *inflated_count = valid_byte_count;
    });
  });
}

//
// TODO
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
  return result;
}

//
// TODO
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
