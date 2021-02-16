//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"


using namespace sycl;

using ProducerToConsumerPipe = INTEL::pipe<  // Defined in the SYCL headers.
    class ProducerConsumerPipe,              // An identifier for the pipe.
    int,                                     // The type of data in the pipe.
    4>;                                      // The capacity of the pipe.

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class ProducerTutorial;
class ConsumerTutorial;

// The Producer kernel reads data from a SYCL buffer and writes it to
// a pipe. This transfers the input data from the host to the Consumer kernel
// that is running concurrently.
event Producer(queue &q, buffer<int, 1> &input_buffer) {
  std::cout << "Enqueuing producer...\n";

  auto e = q.submit([&](handler &h) {
    accessor input_accessor(input_buffer, h, read_only);
    size_t num_elements = input_buffer.get_count();

    h.single_task<ProducerTutorial>([=]() {
      for (size_t i = 0; i < num_elements; ++i) {
        ProducerToConsumerPipe::write(input_accessor[i]);
      }
    });
  });

  return e;
}


// An example of some simple work, to be done by the Consumer kernel
// on the input data
int ConsumerWork(int i) { return i * i; }

// The Consumer kernel reads data from the pipe, performs some work
// on the data, and writes the results to an output buffer
event Consumer(queue &q, buffer<int, 1> &out_buf) {
  std::cout << "Enqueuing consumer...\n";

  auto e = q.submit([&](handler &h) {
    accessor out_accessor(out_buf, h, write_only, noinit);
    size_t num_elements = out_buf.get_count();

    h.single_task<ConsumerTutorial>([=]() {
      for (size_t i = 0; i < num_elements; ++i) {
        // read the input from the pipe
        int input = ProducerToConsumerPipe::read();

        // do work on the input
        int answer = ConsumerWork(input);

        // write the result to the output buffer
        out_accessor[i] = answer;
      }
    });
  });

  return e;
}

int main(int argc, char *argv[]) {
  // Default values for the buffer size is based on whether the target is the
  // FPGA emulator or actual FPGA hardware
#if defined(FPGA_EMULATOR)
  size_t array_size = 1 << 12;
#else
  size_t array_size = 1 << 20;
#endif
  
  // allow the user to change the buffer size at the command line
  if (argc > 1) {
    std::string option(argv[1]);
    if (option == "-h" || option == "--help") {
      std::cout << "Usage: \n./pipes <data size>\n\nFAILED\n";
      return 1;
    } else {
      array_size = atoi(argv[1]);
    }
  }

  std::cout << "Input Array Size: " << array_size << "\n";

  std::vector<int> producer_input(array_size, -1);
  std::vector<int> consumer_output(array_size, -1);

  // Initialize the input data with numbers from 0, 1, 2, ..., array_size-1
  std::iota(producer_input.begin(), producer_input.begin(), 0);

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif

  event producer_event, consumer_event;

  try {
    // property list to enable SYCL profiling for the device queue
    auto props = property_list{property::queue::enable_profiling()};

    // create the device queue with SYCL profiling enabled
    queue q(device_selector, dpc_common::exception_handler, props);

    // create the 
    buffer producer_buffer(producer_input);
    buffer consumer_buffer(consumer_output);

    // Run the two kernels concurrently. The Producer kernel sends
    // data via a pipe to the Consumer kernel.
    producer_event = Producer(q, producer_buffer);
    consumer_event = Consumer(q, consumer_buffer);

  } catch (exception const &e) {
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

  // At this point, the producer_buffer and consumer_buffer have gone out 
  // of scope. This will cause their destructors to be called, which will in 
  // turn block until the Producer and Consumer kernels are finished and the
  // output data is copied back to the host. Therefore, at this point it is
  // safe and correct to access the contents of the consumer_output vector.

  // print profiling information
  // alias the 'info::event_profiling' namespace to save column space
  using syclprof = info::event_profiling;

  // start and end time of the Producer kernel
  double p_start = producer_event.get_profiling_info<syclprof::command_start>();
  double p_end = producer_event.get_profiling_info<syclprof::command_end>();

  // start and end time of the Consumer kernel
  double c_start = consumer_event.get_profiling_info<syclprof::command_start>();
  double c_end = consumer_event.get_profiling_info<syclprof::command_end>();

  // the total application time
  double total_time_ms = (c_end - p_start) * 1e-6;

  // the input size in MBs
  double input_size_mb = array_size * sizeof(int) * 1e-6;

  // the total application throughput
  double throughput_mbs = input_size_mb / (total_time_ms * 1e-3);

  // Print the start times normalized to the start time of the producer.
  // i.e. the producer starts at 0ms and the other start/end times are
  // reported as differences to that number (+X ms).
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "\n";
  std::cout << "Profiling Info\n";
  std::cout << "\tProducer:\n";
  std::cout << "\t\tStart time: " << 0 << " ms\n";
  std::cout << "\t\tEnd time: +" << (p_end-p_start)*1e-6 << " ms\n";
  std::cout << "\t\tKernel Duration: " << (p_end-p_start)*1e-6 << " ms\n";
  std::cout << "\tConsumer:\n";
  std::cout << "\t\tStart time: +" << (c_start-p_start)*1e-6 << " ms\n";
  std::cout << "\t\tEnd time: +" << (c_end-p_start)*1e-6 << " ms\n";
  std::cout << "\t\tKernel Duration: " << (c_end-c_start)*1e-6 << " ms\n";
  std::cout << "\tDesign Duration: " << total_time_ms << " ms\n";
  std::cout << "\tDesign Throughput: " << throughput_mbs << " MB/s\n";
  std::cout << "\n";

  // Verify the result
  for (size_t i = 0; i < array_size; i++) {
    if (consumer_output[i] != ConsumerWork(producer_input[i])) {
      std::cout << "input = " << producer_input[i]
                << " expected: " << ConsumerWork(producer_input[i])
                << " got: " << consumer_output[i] << "\n";
      std::cout << "FAILED: The results are incorrect\n";
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct\n";
  return 0;
}
