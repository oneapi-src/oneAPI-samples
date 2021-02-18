#include <assert.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <functional>
#include <string>
#include <thread>
#include <type_traits>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#include "streaming_without_api.hpp"
#include "streaming_with_api.hpp"

using namespace sycl;

// the type used
// NOTE: the tutorial assumes the use of a sycl::vec datatype (like long8).
// Therefore, 'Type' must be a sycl::vec datatype (e.g. int8, char64, etc).
using Type = long8;

// forward declare the roofline analysis function
template<typename T>
void DoRooflineAnalysis(queue& q, size_t buffer_count, size_t iterations,
                        size_t threads);

// the main function
int main(int argc, char* argv[]) {
  // parse command line arguments
#if defined(FPGA_EMULATOR)
  size_t reps = 20;
  size_t buffer_count = 1 << 12;  // 4096
  size_t iterations = 2;
#else
  size_t reps = 200;
  size_t buffer_count = 1 << 19;  // 524388
  size_t iterations = 5;
#endif

  // auto-detect the number of available threads
  size_t detected_threads = (size_t)(std::thread::hardware_concurrency());

  // thread::hardware_concurrency() returns 0 if it cannot determine
  // the number of threads, so fallback to 2
  size_t threads = std::max(size_t(2), detected_threads);

  size_t buffers = 2;
  bool need_help = false;

  // parse the command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      need_help = true;
    } else {
      std::string str_after_equals = arg.substr(arg.find("=") + 1);

      if (arg.find("--reps=") == 0) {
        reps = atoi(str_after_equals.c_str());
      } else if (arg.find("--buffers=") == 0) {
        buffers = atoi(str_after_equals.c_str());
      } else if (arg.find("--buffer_count=") == 0) {
        buffer_count = atoi(str_after_equals.c_str());
      } else if (arg.find("--iterations=") == 0) {
        iterations = std::max(2, atoi(str_after_equals.c_str()) + 1);
      } else if (arg.find("--threads=") == 0) {
        threads = atoi(str_after_equals.c_str());
      }  else {
        std::cout << "WARNING: ignoring unknown argument '" << arg << "'\n";
      }
    }
  }

  // print help is asked
  if (need_help) {
    std::cout << "USAGE: "
              << "./buffered_host_streaming "
              << "[--reps=<int>] "
              << "[--buffers=<int>] "
              << "[--buffer_count=<int>] "
              << "[--iterations=<int>] "
              << "[--threads=<int>]\n";
    return 0;
  }

  // check the reps
  if (reps <= 0) {
    std::cerr << "ERROR: 'reps' must be greater than 0\n";
    std::terminate();
  }

  // check the buffer_count
  if (buffer_count <= 0) {
    std::cerr << "ERROR: 'buffer_count' must be greater than 0\n";
    std::terminate();
  }

  // check the number of iterations
  if (iterations <= 0) {
    std::cerr << "ERROR: 'iterations' must be positive\n";
    std::terminate();
  }

  if (threads <= 0) {
    std::cerr << "ERROR: 'threads' must be positive\n";
    std::terminate();
  }

  // print info
  std::cout << "Repetitions:      " << reps << "\n";
  std::cout << "Buffers:          " << buffers << "\n";
  std::cout << "Buffer Count:     " << buffer_count << "\n";
  std::cout << "Iterations:       " << iterations-1 << "\n";
  std::cout << "Total Threads:    " << threads << "\n";
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

    ///////////////////////////////////////////////////////////////////////////
    // find the bandwidth of each processing component in our design
    std::cout << "Running the roofline analysis\n";
    DoRooflineAnalysis<Type>(q, buffer_count, iterations, threads);
    std::cout << "Done the roofline analysis\n\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // run the design that does not use the API (see streaming_without_api.hpp)
    std::cout << "Running the full design without API\n";
    passed &= DoWork<Type>(q, buffers, buffer_count, reps, iterations, threads);
    std::cout << "\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // run the design that uses the API (see streaming_with_api.hpp)
    std::cout << "Running the full design with API\n";
    passed &= DoWorkAPI<Type>(q, buffers, buffer_count, reps, iterations,
                              threads);
    std::cout << "\n";
    ///////////////////////////////////////////////////////////////////////////

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

// This function performs a bandwidth test on the individual components of the
// processing pipeline: Producer, Consumer, and the FPGA kernel.
// It computes (and returns) the maximum possible throughput of the full design
// when the individual components are combined. 
template<typename T>
void DoRooflineAnalysis(queue& q, size_t buffer_count, size_t iterations,
                        size_t threads) {
  // allocate some memory to work with
  T *tmp_in[2], *tmp_out[2];
  for (size_t i = 0; i < 2; i++) {
    if ((tmp_in[i] = malloc_host<T>(buffer_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for "
                << "'tmp_in[" << i << "]'\n";
      std::terminate();
    }
    if ((tmp_out[i] = malloc_host<T>(buffer_count, q)) == nullptr) {
      std::cerr << "ERROR: could not allocate space for "
                << "'tmp_out[" << i << "]'\n";
      std::terminate();
    }
  }

  // the Producer and Consumer get half of the total threads, each.
  // std::max() guards against the case where there is only 1 thread.
  size_t half_threads = std::max(size_t(1), threads/2);

  // these tests are quick, so run some extra iterations for more accuracy
  size_t bw_test_iterations = iterations * 4 + 1;

  // timing variables
  high_resolution_clock::time_point start, end;
  duration<double, std::milli> delta_ms;
  double processing_time_producer = 0.0, tp_producer;
  double processing_time_consumer = 0.0, tp_consumer;
  double processing_time_producer_consumer = 0.0, tp_producer_consumer;
  double processing_time_kernel = 0.0, tp_kernel;

  // the total number of megabytes processed by the operations
  double size_mb = sizeof(T) * buffer_count * 1e-6;

  // do multiple interations of the test to improve measurement accuracy
  for (size_t i = 0; i < bw_test_iterations; i++) {
    // generate some data
    std::fill_n(tmp_out[0], buffer_count, i);
    std::fill_n(tmp_out[1], buffer_count, i);

    // Producer in isolation
    start = high_resolution_clock::now();
    // ProducerFunction is defined in common.hpp
    ProducerFunction(tmp_out[0], tmp_in[0], buffer_count, half_threads);
    end = high_resolution_clock::now();
    delta_ms = end - start;
    if (i > 0) processing_time_producer += delta_ms.count();

    // wait 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Consumer in isolation
    start = high_resolution_clock::now();
    // ConsumerFunction is defined in common.hpp
    ConsumerFunction(tmp_in[0], tmp_out[0], buffer_count, half_threads);
    end = high_resolution_clock::now();
    delta_ms = end - start;
    if (i > 0) processing_time_consumer += delta_ms.count();

    // wait 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Producer & Consumer at the same time
    // NOTE: this is the most accurate measurement of our actual design,
    // since the Producer and Consumer will be executing in parallel (see
    // README.md for more details on this).
    // The ProducerFunction and ConsumerFunction are defined in common.hpp
    start = high_resolution_clock::now();
    std::thread producer_thread([&] {
      ProducerFunction(tmp_out[0], tmp_in[0], buffer_count, half_threads);
    });
    ConsumerFunction(tmp_out[1], tmp_in[1], buffer_count, half_threads);
    producer_thread.join();
    end = high_resolution_clock::now();
    delta_ms = end - start;
    if (i > 0) processing_time_producer_consumer += delta_ms.count();

    // wait 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Kernel in isolation
    start = high_resolution_clock::now();
    // SubmitKernel is defined in streaming_without_api.hpp
    auto e = SubmitKernel(q, tmp_in[0], buffer_count, tmp_out[0]);
    e.wait();
    end = high_resolution_clock::now();
    delta_ms = end - start;
    if (i > 0) processing_time_kernel += delta_ms.count();

    // wait 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // average the processing times across iterations
  processing_time_producer /= (bw_test_iterations-1);
  processing_time_consumer /= (bw_test_iterations-1);
  processing_time_producer_consumer /= (bw_test_iterations-1);
  processing_time_kernel /= (bw_test_iterations-1);

  // compute throughputs
  tp_producer = size_mb / (processing_time_producer * 1e-3);
  tp_consumer = size_mb / (processing_time_consumer * 1e-3);
  tp_producer_consumer = size_mb / (processing_time_producer_consumer * 1e-3);
  tp_kernel = size_mb / (processing_time_kernel * 1e-3);

  // print results
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Producer (" << half_threads << " threads)\n";
  std::cout << "\tTime:       " << processing_time_producer << " ms\n";
  std::cout << "\tThroughput: " << tp_producer << " MB/s\n";

  std::cout << "Consumer (" << half_threads << " threads)\n";
  std::cout << "\tTime:       " << processing_time_consumer << " ms\n";
  std::cout << "\tThroughput: " << tp_consumer << " MB/s\n";

  std::cout << "Producer & Consumer (" << half_threads << " threads, each)\n";
  std::cout << "\tTime:       " << processing_time_producer_consumer << " ms\n";
  std::cout << "\tThroughput: " << tp_producer_consumer << " MB/s\n";

  std::cout << "Kernel\n";
  std::cout << "\tTime:       " << processing_time_kernel << " ms\n";
  std::cout << "\tThroughput: " << tp_kernel << " MB/s\n";

  // find the minimum throughput (which will bottleneck the design)
  std::vector<double> tps = {tp_producer,
                             tp_consumer,
                             tp_producer_consumer,
                             tp_kernel};
  
  // the index of the min throughput
  int min_tp_idx = std::min_element(tps.begin(), tps.end()) - tps.begin();

  // the minimum throughput
  double min_tp = tps[min_tp_idx];

  // check if the bottleneck throughput is the kernel
  bool kernel_is_limit = (min_tp_idx == tps.size()-1);

  // the minimum throughput is the maximum throughput of the full design
  std::cout << "\n";
  std::cout << "Maximum Design Throughput: " << min_tp << " MB/s\n";
  std::cout << "The FPGA kernel "
            << (kernel_is_limit ? "limits" : "does not limit")
            << " the performance of the design\n";

  // free temp USM memory
  for (size_t i = 0; i < 2; i++) {
    sycl::free(tmp_in[i], q);
    sycl::free(tmp_out[i], q);
  }
}
