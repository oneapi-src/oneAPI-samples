#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#include "single_kernel.hpp"
#include "multi_kernel.hpp"

using namespace sycl;
using namespace std::chrono;

// data types and constants
// NOTE: this tutorial assumes you are using a sycl::vec datatype. Therefore, 
// 'Type' can only be changed to a different vector datatype (e.g. int16,
// ulong8, etc...)
using Type = long8;

///////////////////////////////////////////////////////////////////////////////
// forward declaration of the functions in this file
// the function definitions are all below the main() function in this file
template<typename T>
void DoWorkOffload(queue& q, T* in, T* out, size_t total_count,
                   size_t iterations);

template<typename T>
void DoWorkSingleKernel(queue& q, T* in, T* out,
                        size_t chunks, size_t chunk_count, size_t total_count,
                        size_t inflight_kernels, size_t iterations);

template <typename T>
void DoWorkMultiKernel(queue& q, T* in, T* out,
                       size_t chunks, size_t chunk_count, size_t total_count,
                       size_t inflight_kernels, size_t iterations);

template<typename T>
void PrintPerformanceInfo(std::string print_prefix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms);
///////////////////////////////////////////////////////////////////////////////


int main(int argc, char* argv[]) {
  // default values
#if defined(FPGA_EMULATOR)
  size_t chunks = 1 << 4;         // 16
  size_t chunk_count = 1 << 8;    // 256
  size_t iterations = 2;
#else
  size_t chunks = 1 << 9;         // 512
  size_t chunk_count = 1 << 15;   // 32768
  size_t iterations = 5;
#endif

  // This is the number of kernels we will have in the queue at a single time.
  // If this number is set too low (e.g. 1) then we don't take advantage of
  // fast kernel relaunch (see the README). If this number is set to high,
  // then the first kernel launched finishes before we are done launching all
  // the kernels and therefore throughput is decreased.
  size_t inflight_kernels = 2;

  // parse the command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      std::cout << "USAGE: "
                << "./simple_host_streaming "
                << "[--chunks=<int>] "
                << "[--chunk_count=<int>] "
                << "[--inflight_kernels=<int>] "
                << "[--iterations=<int>]\n";
      return 0;
    } else {
      std::string str_after_equals = arg.substr(arg.find("=") + 1);

      if (arg.find("--chunks=") == 0) {
        chunks = atoi(str_after_equals.c_str());
      } else if (arg.find("--chunk_count=") == 0) {
        chunk_count = atoi(str_after_equals.c_str());
      } else if (arg.find("--inflight_kernels=") == 0) {
        inflight_kernels = atoi(str_after_equals.c_str());
      } else if (arg.find("--iterations=") == 0) {
        iterations = std::max(2, atoi(str_after_equals.c_str()) + 1);
      } else {
        std::cout << "WARNING: ignoring unknown argument '" << arg << "'\n";
      }
    }
  }

  // check the chunks
  if (chunks <= 0) {
    std::cerr << "ERROR: 'chunks' must be greater than 0\n";
    std::terminate();
  }

  // check the chunk size
  if (chunk_count <= 0) {
    std::cerr << "ERROR: 'chunk_count' must be greater than 0\n";
    std::terminate();
  }

  // check inflight_kernels
  if (inflight_kernels <= 0) {
    std::cerr << "ERROR: 'inflight_kernels' must be positive\n";
    std::terminate();
  }

  // check the number of iterations
  if (iterations <= 0) {
    std::cerr << "ERROR: 'iterations' must be positive\n";
    std::terminate();
  }

  // compute the total number of elements
  size_t total_count = chunks * chunk_count;

  std::cout << "# Chunks:             " << chunks << "\n";
  std::cout << "Chunk count:          " << chunk_count << "\n";
  std::cout << "Total count:          " << total_count << "\n";
  std::cout << "Iterations:           " << iterations-1 << "\n";
  std::cout << "\n";

  bool passed = true;

  try {
    // device selector
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector selector;
#else
    INTEL::fpga_selector selector;
#endif

    // queue properties to enable profiling
    property_list prop_list { property::queue::enable_profiling() };

    // create the device queue
    queue q(selector, dpc_common::exception_handler, prop_list);

    // make sure the device supports USM host allocations
    device d = q.get_device();
    if (!d.get_info<info::device::usm_host_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
      std::terminate();
    }

    // the USM input and output data
    Type *in, *out;
    if ((in = malloc_host<Type>(total_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'in'\n";
      std::terminate();
    }
    if ((out = malloc_host<Type>(total_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'out'\n";
      std::terminate();
    }

    // generate the random input data
    // NOTE: by generating all of the data ahead of time, we are essentially
    // assuming that the producer of data (producing data for the FPGA to
    // consume) has infinite bandwidth. However, if the producer of data cannot
    // produce data faster than our FPGA can consume it, the CPU producer will
    // bottleneck the total throughput of the design.
    std::generate_n(in, total_count, [] { return Type(rand() % 100); });

    // a lambda function to validate the results
    auto validate_results = [&] {
      for (size_t i = 0; i < total_count; i++) {
        auto comp = (in[i] == out[i]);
        for (auto j = 0; j < comp.get_count(); j++) {
          if (!comp[j]) {
            std::cerr << "ERROR: Values do not match, "
                      << "in[" << i << "][" << j << "]:" << in[i][j]
                      << " != out[" << i << "]["<< j << "]:" << out[i][j]
                      << "\n";
            return false;
          }
        }
      }

      return true;
    };

    ////////////////////////////////////////////////////////////////////////////
    // run the offload version, which is NOT optimized for latency at all
    std::cout << "Running the basic offload kernel\n";
    DoWorkOffload(q, in, out, total_count, iterations);

    // validate the results using the lambda
    passed &= validate_results();

    std::cout << "\n";
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // run the optimized (for latency) version that uses fast kernel relaunch
    // by keeping at most 'inflight_kernels' in the SYCL queue at a time
    std::cout << "Running the latency optimized single-kernel design\n";
    DoWorkSingleKernel(q, in, out, chunks, chunk_count, total_count,
                       inflight_kernels, iterations);

    // validate the results using the lambda
    passed &= validate_results();

    std::cout << "\n";
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // run the optimized (for latency) version with multiple kernels that uses
    // fast kernel relaunch by keeping at most 'inflight_kernels' in the SYCL
    // queue at a time
    std::cout << "Running the latency optimized multi-kernel design\n";
    DoWorkMultiKernel(q, in, out, chunks, chunk_count, total_count,
                      inflight_kernels, iterations);

    // validate the results using the lambda
    passed &= validate_results();

    std::cout << "\n";
    ////////////////////////////////////////////////////////////////////////////

    // free the USM pointers
    sycl::free(in, q);
    sycl::free(out, q);

  } catch (exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  if(passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}

// the basic offload kernel version (doesn't care about latency)
template<typename T>
void DoWorkOffload(queue& q, T* in, T* out, size_t total_count,
                   size_t iterations) {
  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  for (size_t i = 0; i < iterations; i++) {
    auto start = high_resolution_clock::now();

    // submit single kernel for entire buffer
    // this function is defined in 'single_kernel.hpp'
    auto e = SubmitSingleWorker(q, in, out, total_count);

    // wait on the kernel to finish
    e.wait();

    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> process_time = end - start;

    // in offload designs, the processing time and latency are identical
    // since the synchronization between the host and device is coarse grain
    // (i.e. the synchronization happens once ALL the data has been processed).
    latency_ms[i] = process_time.count();
    process_time_ms[i] = process_time.count();
  }

  // compute and print timing information
  PrintPerformanceInfo<T>("Offload",
                          total_count, latency_ms, process_time_ms);
}

// The single-kernel version of the design.
// This function optimizes for latency (while maintaining throughput) by
// breaking the computation into 'chunks' and launching kernels for each
// chunk. The synchronization of the kernel ending tells the host that the data
// for the given chunk is ready in the output buffer.
template <typename T>
void DoWorkSingleKernel(queue& q, T* in, T* out,
                        size_t chunks, size_t chunk_count, size_t total_count,
                        size_t inflight_kernels, size_t iterations) {
  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  // count the number of chunks for which kernels have been started
  size_t in_chunk = 0;

  // count the number of chunks for which kernels have finished 
  size_t out_chunk = 0;

  // use a queue to track the kernels in flight
  // By queueing multiple kernels before waiting on the oldest to finish
  // (inflight_kernels) we still have kernels in the SYCL queue and ready to
  // launch while we call event.wait() on the oldest kernel in the queue.
  // However, if we set 'inflight_kernels' too high, then the time to launch
  // the first set of kernels will be longer than the time for the first kernel
  // to finish and our latency and throughput will be negatively affected.
  std::queue<event> event_q;

  for (size_t i = 0; i < iterations; i++) {
    // reset the output data to catch any untouched data
    std::fill_n(out, total_count, -1);

    // reset counters
    in_chunk = 0;
    out_chunk = 0;

    // clear the queue
    std::queue<event> clear_q;
    std::swap(event_q, clear_q);

    // latency timers
    high_resolution_clock::time_point first_data_in, first_data_out;

    auto start = high_resolution_clock::now();

    do {
      // if we still have kernels to launch, launch them in here
      if (in_chunk < chunks) {
        // launch the kernel
        size_t chunk_offset = in_chunk*chunk_count; 
        // this function is defined in 'single_kernel.hpp'
        auto e = SubmitSingleWorker(q, in + chunk_offset, out + chunk_offset,
                                    chunk_count);

        // push the kernel event into the queue
        event_q.push(e);

        // if this is the first chunk, track the time
        if (in_chunk == 0) first_data_in = high_resolution_clock::now();
        in_chunk++;
      }

      // wait on the earliest kernel to finish if either condition is met:
      //    1) there are a certain number kernels in flight
      //    2) all of the kernels have been launched
      if ((event_q.size() >= inflight_kernels) || (in_chunk >= chunks)) {
        // pop the earliest kernel event we are waiting on
        auto e = event_q.front();
        event_q.pop();

        // wait on it to finish
        e.wait();

        // track the time if this is the first producer/consumer pair
        if (out_chunk == 0) first_data_out = high_resolution_clock::now();

        // The synchronization of the kernels ending tells us that, at this 
        // point, the first 'out_chunk' chunks are valid on the host.
        // NOTE: This is the point where you would consume the output data
        // at (out + out_chunk*chunk_size).
        out_chunk++;
      }
    } while (out_chunk < chunks);

    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> latency = first_data_out - first_data_in;
    duration<double, std::milli> process_time = end - start;
    latency_ms[i] = latency.count();
    process_time_ms[i] = process_time.count();
  }

  // compute and print timing information
  PrintPerformanceInfo<T>("Single-kernel",
                          total_count, latency_ms, process_time_ms);
}

//
// The multi-kernel version of the design.
// Like the single-kernel version of the design, this design optimizes for 
// latency (while maintaining throughput) by breaking the producing and
// consuming of data into chunks. That is, the main kernel pipeline (K0, 
// K1, and K2 from SubmitMultiKernelWorkers above) are enqueued ONCE but
// the producer and consumer kernels, that feed and consume data to the
// the kernel pipeline, are broken into smaller chunks. The synchronization of
// the producer and consumer kernels (specifically, the consumer kernel)
// signals to the host that a new chunk of data is ready in host memory.
// See the README file for more information on why a producer and consumer
// kernel are created for this design style.
//
// The following is a block diagram of this kernel this function creates:
//
//  in |---| ProducePipe |----| Pipe0 |----| Pipe1 |----| ConsumePipe |---| out
// --->| P |============>| K0 |======>| K1 |======>| K2 |============>| C |---->
//     |---|             |----|       |----|       |----|             |---|
//

// the pipes used to produce/consume data
using ProducePipe = pipe<class ProducePipeClass, Type>;
using ConsumePipe = pipe<class ConsumePipeClass, Type>;

template <typename T>
void DoWorkMultiKernel(queue& q, T* in, T* out,
                        size_t chunks, size_t chunk_count, size_t total_count,
                        size_t inflight_kernels, size_t iterations) {
  // timing data
  std::vector<double> latency_ms(iterations);
  std::vector<double> process_time_ms(iterations);

  // count the number of chunks for which kernels have been started
  size_t in_chunk = 0;

  // count the number of chunks for which kernels have finished 
  size_t out_chunk = 0;

  // use a queue to track the kernels in flight
  std::queue<std::pair<event,event>> event_q;

  for (size_t i = 0; i < iterations; i++) {
    // reset the output data to catch any untouched data
    std::fill_n(out, total_count, -1);

    // reset counters
    in_chunk = 0;
    out_chunk = 0;

    // clear the queue
    std::queue<std::pair<event,event>> clear_q;
    std::swap(event_q, clear_q);

    // latency timers
    high_resolution_clock::time_point first_data_in, first_data_out;

    // launch the worker kernels
    // NOTE: these kernels will process ALL of the data (total_count)
    // while the producer/consumer will be broken into chunks
    // this function is defined in 'multi_kernel.hpp'
    auto events = SubmitMultiKernelWorkers<T,
                                           ProducePipe,
                                           ConsumePipe>(q, total_count);

    auto start = high_resolution_clock::now();

    do {
      // if we still have kernels to launch, launch them in here
      if (in_chunk < chunks) {
        // launch the producer/consumer pair for the next chunk of data
        size_t chunk_offset = in_chunk*chunk_count;

        // these functions are defined in 'multi_kernel.hpp'
        event p_e = SubmitProducer<T, ProducePipe>(q, in + chunk_offset,
                                                   chunk_count);
        event c_e = SubmitConsumer<T, ConsumePipe>(q, out + chunk_offset,
                                                   chunk_count);

        // push the kernel event into the queue
        event_q.push(std::make_pair(p_e, c_e));

        // if this is the first chunk, track the time
        if (in_chunk == 0) first_data_in = high_resolution_clock::now();
        in_chunk++;
      }

      // wait on the oldest kernel to finish if any of these conditions are met:
      //    1) there are a certain number kernels in flight
      //    2) all of the kernels have been launched
      //
      // NOTE: 'inflight_kernels' is now the number of inflight
      // producer/consumer kernel pairs
      if ((event_q.size() >= inflight_kernels) || (in_chunk >= chunks)) {
        // grab the oldest kernel event we are waiting on
        auto event_pair = event_q.front();
        event_q.pop();

        // wait on the producer/consumer kernel pair to finish
        event_pair.first.wait();    // producer
        event_pair.second.wait();   // consumer

        // track the time if this is the first producer/consumer pair
        if (out_chunk == 0) first_data_out = high_resolution_clock::now();

        // at this point the first 'out_chunk' chunks are ready to be
        // processed on the host
        out_chunk++;
      }
    } while(out_chunk < chunks);

    // wait for the worker kernels to finish, which should be done quickly
    // since all producer/consumer pairs are done
    for (auto& e : events) {
      e.wait();
    }

    auto end = high_resolution_clock::now();

    // compute latency and processing time
    duration<double, std::milli> latency = first_data_out - first_data_in;
    duration<double, std::milli> process_time = end - start;
    latency_ms[i] = latency.count();
    process_time_ms[i] = process_time.count();
  }

  // compute and print timing information
  PrintPerformanceInfo<T>("Multi-kernel",
                          total_count, latency_ms, process_time_ms);
}

// a helper function to compute and print the performance info
template<typename T>
void PrintPerformanceInfo(std::string print_prefix, size_t count,
                          std::vector<double>& latency_ms,
                          std::vector<double>& process_time_ms) {
  // compute the input size in MB
  double input_size_megabytes = (sizeof(T) * count) * 1e-6;

  // compute the average latency and processing time
  double iterations = latency_ms.size() - 1;
  double avg_latency_ms = std::accumulate(latency_ms.begin() + 1,
                                          latency_ms.end(),
                                          0.0) / iterations;
  double avg_processing_time_ms = std::accumulate(process_time_ms.begin() + 1,
                                                  process_time_ms.end(),
                                                  0.0) / iterations;

  // compute the throughput
  double avg_tp_mb_s = input_size_megabytes / (avg_processing_time_ms * 1e-3);

  // print info
  std::cout << std::fixed << std::setprecision(4);
  std::cout << print_prefix
            << " average latency:           " << avg_latency_ms << " ms\n";
  std::cout << print_prefix
            << " average throughput:        " << avg_tp_mb_s  << " MB/s\n";
}
