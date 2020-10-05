//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

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

using ProducerToConsumerPipe = INTEL::pipe<  // Defined in the SYCL headers.
    class ProducerConsumerPipe,              // An identifier for the pipe.
    int,                                     // The type of data in the pipe.
    4>;                                      // The capacity of the pipe.

// Forward declare the kernel names
// (This will become unnecessary in a future compiler version.)
class ProducerTutorial;
class ConsumerTutorial;

// The Producer kernel reads data from a SYCL buffer and writes it to
// a pipe. This transfers the input data from the host to the Consumer kernel
// that is running concurrently.
void Producer(queue &q, buffer<int, 1> &input_buffer) {
  std::cout << "Enqueuing producer...\n";

  auto e = q.submit([&](handler &h) {
    auto input_accessor = input_buffer.get_access<access::mode::read>(h);
    size_t num_elements = input_buffer.get_count();

    h.single_task<ProducerTutorial>([=]() {
      for (size_t i = 0; i < num_elements; ++i) {
        ProducerToConsumerPipe::write(input_accessor[i]);
      }
    });
  });
}


// An example of some simple work, to be done by the Consumer kernel
// on the input data
int ConsumerWork(int i) { return i * i; }

// The Consumer kernel reads data from the pipe, performs some work
// on the data, and writes the results to an output buffer
void Consumer(queue &q, buffer<int, 1> &out_buf) {
  std::cout << "Enqueuing consumer...\n";

  auto e = q.submit([&](handler &h) {
    auto out_accessor = out_buf.get_access<access::mode::discard_write>(h);
    size_t num_elements = out_buf.get_count();

    h.single_task<ConsumerTutorial>([=]() {
      for (size_t i = 0; i < num_elements; ++i) {
        int input = ProducerToConsumerPipe::read();
        int answer = ConsumerWork(input);
        out_accessor[i] = answer;
      }
    });
  });
}

int main(int argc, char *argv[]) {
  size_t array_size = (1 << 10);

  if (argc > 1) {
    std::string option(argv[1]);
    if (option == "-h" || option == "--help") {
      std::cout << "Usage: \n<executable> <data size>\n\nFAILED\n";
      return 1;
    } else {
      array_size = std::stoi(option);
    }
  }

  std::cout << "Input Array Size:  " << array_size << "\n";

  std::vector<int> producer_input(array_size, -1);
  std::vector<int> consumer_output(array_size, -1);

  // Initialize the input data
  for (size_t i = 0; i < array_size; i++)
    producer_input[i] = i;

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif

  try {
    queue q(device_selector, dpc_common::exception_handler);

    buffer producer_buffer(producer_input);
    // Use verbose SYCL 1.2 syntax for the output buffer.
    // (This will become unnecessary in a future compiler version.)
    buffer<int, 1> consumer_buffer(consumer_output.data(), array_size);

    // Run the two kernels concurrently. The Producer kernel sends
    // data via a pipe to the Consumer kernel.
    Producer(q, producer_buffer);
    Consumer(q, consumer_buffer);

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

  // Verify result
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
