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

#include "Decompressor.hpp"
#include "HeaderData.hpp"
#include "ByteHistory.hpp"

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

using InPipe = ext::intel::pipe<InPipeID, char>;
using OutPipe = ext::intel::pipe<OutPipeID, FlagBundle<unsigned char>>;

// Max compression ratio. For example, 5 means the decompressed file is 5x
// bigger than the compressed file.
constexpr int kInflateFactor = 5;

////////////////////////////////////////////////////////////////////////////////
event SubmitProducer(queue&, unsigned char*, int);
event SubmitConsumer(queue&, unsigned char*, int*, int);
std::vector<unsigned char> ReadInputFile(std::string filename);
void WriteOutputFile(std::string, std::vector<unsigned char>&);
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
  bool passed = true;

  // reading and validating the command line arguments
  std::string in_filename = "../data/in.gz";
  std::string out_filename = "../data/out";
  
#ifdef FPGA_EMULATOR
  //int runs = 2;
  int runs = 1;
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
  /*
  if (runs < 2) {
    std::cerr << "ERROR: 'runs' must be 2 or more\n";
    std::terminate();
  }
  */

  // the device selector
#ifdef FPGA_EMULATOR
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  // create the device queue
  queue q(selector, dpc_common::exception_handler);

  // read input file
  std::vector<unsigned char> in_bytes = ReadInputFile(in_filename);
  int in_count = in_bytes.size();
  int max_out_count = in_count * kInflateFactor;
  std::vector<unsigned char> out_bytes(max_out_count);
  int inflated_count_host = 0;
  HeaderData hdr_data_host;
  unsigned int crc_host, size_host;

  std::cout << "in_count = " << in_count << " bytes\n";

  // track timing information in ms
  std::vector<double> time(runs);

  // input and output data pointers on the device using USM device allocations
  unsigned char *in, *out;
  int *inflated_count ;
  HeaderData *hdr_data;
  int *crc;
  int *size;

  try {
    // allocate memory on the device for the input and output
    if ((in = malloc_device<unsigned char>(in_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out = malloc_device<unsigned char>(max_out_count, q)) == nullptr) {
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
    if ((size = malloc_device<int>(1, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'size'\n";
      std::terminate();
    }

    // copy the input data to the device memory and wait for the copy to finish
    q.memcpy(in, in_bytes.data(), in_count * sizeof(unsigned char)).wait();

    // run the design multiple times to increase the accuracy of the timing
    for (int i = 0; i < runs; i++) {
      // run the producer and consumer kernels
      auto producer_event = SubmitProducer(q, in, in_count);
      auto consumer_event = SubmitConsumer(q, out, inflated_count, max_out_count);

      // run the decompression kernels
      auto header_event = SubmitHeaderKernel<InPipe, HeaderToHuffmanPipe>(q, hdr_data, in_count, crc, size);
      auto huffman_event = SubmitHuffmanDecoderKernel<HeaderToHuffmanPipe, HuffmanToLZ77Pipe>(q);
      auto lz77_kernel = SubmitLZ77DecoderKernel<HuffmanToLZ77Pipe, OutPipe>(q);

      // wait for the producer and consumer to finish
      auto start = high_resolution_clock::now();
      producer_event.wait();
      //std::cout << "producer_event\n";
      consumer_event.wait();
      //std::cout << "consumer_event\n";
      auto end = high_resolution_clock::now();

      // wait for the decompression kernels to finish
      header_event.wait();
      //std::cout << "header_event\n";
      huffman_event.wait();
      //std::cout << "huffman_event\n";
      lz77_kernel.wait();
      //std::cout << "lz77_kernel\n";

      // calculate the time the kernels ran for, in milliseconds
      time[i] = duration<double, std::milli>(end - start).count();

      // Copy the output back from the device
      q.memcpy(out_bytes.data(), out, max_out_count * sizeof(unsigned char)).wait();
      q.memcpy(&inflated_count_host, inflated_count, sizeof(int)).wait();
      q.memcpy(&hdr_data_host, hdr_data, sizeof(HeaderData)).wait();
      q.memcpy(&crc_host, crc, sizeof(int)).wait();
      q.memcpy(&size_host, size, sizeof(int)).wait();

      // validate the results
      // keep the first inflated_count_host bytes
      out_bytes.resize(inflated_count_host);
      assert(inflated_count_host == size_host);

      // TODO
      std::cout << "inflated_count_host = " << inflated_count_host << " bytes\n";
      std::cout << hdr_data_host;
      std::cout << "crc = " << crc_host << "\n";
      std::cout << "size = " << size_host << "\n";
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
  WriteOutputFile(out_filename, out_bytes);

  // print the performance results
  if (passed) {
    // NOTE: when run in emulation, these results do not accurately represent
    // the performance of the kernels in actual FPGA hardware
    double avg_time_ms =
        std::accumulate(time.begin() + 1, time.end(), 0.0) / (runs - 1);

    // TODO: what should the count be here? Input size? Output size?
    size_t megabytes = in_count * sizeof(unsigned char) * 1e-6;

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
event SubmitConsumer(queue& q, unsigned char* out_ptr, int* inflated_count_ptr, int max_count) {
  return q.submit([&](handler &h) {
    h.single_task<ConsumerID>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<unsigned char> out(out_ptr);
      device_ptr<int> inflated_count(inflated_count_ptr);

      int i = 0;
      bool done;
      do {
        // read the pipe data
        bool valid_pipe_read;
        auto pipe_data = OutPipe::read(valid_pipe_read);
        done = pipe_data.flag && valid_pipe_read;

        if (!done && valid_pipe_read) {
          out[i] = pipe_data.data;
          i++;
        }
      } while (i < max_count && !done);

      // write out the actual output count (inflated_count <= max_count)
      *inflated_count = i;
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
