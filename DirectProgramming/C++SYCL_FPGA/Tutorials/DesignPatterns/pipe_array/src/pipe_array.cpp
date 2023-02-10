//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"

#include "exception_handler.hpp"

using namespace sycl;

constexpr size_t kNumRows = 2;
constexpr size_t kNumCols = 2;
constexpr size_t kNumberOfConsumers = kNumRows * kNumCols;
constexpr size_t kDepth = 2;

using ProducerToConsumerPipeMatrix =
    fpga_tools::PipeArray<          // Defined in "pipe_utils.hpp".
      class ProducerConsumerPipe,   // An identifier for the pipe.
      uint64_t,                     // The type of data in the pipe.
      kDepth,                       // The capacity of each pipe.
      kNumRows,                     // array dimension.
      kNumCols                      // array dimension.
    >;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class ProducerTutorial;
template <size_t consumer_id> class ConsumerTutorial;

void Producer(queue &q, buffer<uint64_t, 1> &input_buffer) {
  std::cout << "Enqueuing producer...\n";

  auto e = q.submit([&](handler &h) {
    accessor in(input_buffer, h, read_only);
    auto num_elements = input_buffer.size();
    auto num_passes = num_elements / kNumberOfConsumers;

    // The producer kernel writes to every pipe in the 2D pipe array
    h.single_task<ProducerTutorial>([=]() {
      size_t input_idx = 0;
      for (size_t pass = 0; pass < num_passes; pass++) {
        // Template-based unroll (outer "i" loop)
        fpga_tools::UnrolledLoop<kNumRows>([&input_idx, in](auto i) {
          // Template-based unroll (inner "j" loop)
          fpga_tools::UnrolledLoop<kNumCols>([&input_idx, &i, in](auto j) {
            // Write a value to the <i,j> pipe of the pipe array
            ProducerToConsumerPipeMatrix::PipeAt<i, j>::write(in[input_idx++]);
          });
        });
      }
    });
  });
}

// Do some work on the data (any function could be substituted)
uint64_t ConsumerWork(uint64_t i) { return i * i; }

template <size_t consumer_id>
void Consumer(queue &q, buffer<uint64_t, 1> &out_buf) {
  std::cout << "Enqueuing consumer " << consumer_id << "...\n";

  auto e = q.submit([&](handler &h) {
    accessor out(out_buf, h, write_only, no_init);
    auto num_elements = out_buf.size();

    // The consumer kernel reads from a single pipe, determined by consumer_id
    h.single_task<ConsumerTutorial<consumer_id>>([=]() {
      constexpr size_t x = consumer_id / kNumCols;
      constexpr size_t y = consumer_id % kNumCols;
      for (size_t i = 0; i < num_elements; ++i) {
        auto input = ProducerToConsumerPipeMatrix::PipeAt<x, y>::read();
        out[i] = ConsumerWork(input);
      }
    });
  });
}

int main(int argc, char *argv[]) {
  uint64_t array_size = 1;
  array_size <<= 10;

  // Parse optional data size argument
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

  // Check input validity
  if (array_size % kNumberOfConsumers != 0) {
    std::cout << "Array size must be a multiple of the number of consumers! "
                 "Exiting...\n";
    return 0;
  }

  // Set up producer input vector, and kNumberOfConsumers output vectors
  uint64_t items_per_consumer = array_size / kNumberOfConsumers;
  std::vector<uint64_t> producer_input(array_size, -1);
  std::array<std::vector<uint64_t>, kNumberOfConsumers> consumer_output;

  for (auto &output : consumer_output)
    output.resize(items_per_consumer, -1);

  // Initialize producer input
  for (size_t i = 0; i < array_size; i++)
    producer_input[i] = i;

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    queue q(selector, fpga_tools::exception_handler);

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Enqueue producer
    buffer<uint64_t,1> producer_buffer(producer_input);
    Producer(q, producer_buffer);

    std::vector<buffer<uint64_t,1>> consumer_buffers;

    // Use template-based unroll to enqueue multiple consumers
    fpga_tools::UnrolledLoop<kNumberOfConsumers>([&](auto consumer_id) {
      consumer_buffers.emplace_back(consumer_output[consumer_id].data(),
                                    items_per_consumer);
      Consumer<consumer_id>(q, consumer_buffers.back());
    });

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
    }
    std::terminate();
  }

  // Verify result
  for (size_t i = 0; i < items_per_consumer; ++i) {
    for (size_t consumer = 0; consumer < kNumberOfConsumers; ++consumer) {
      auto fpga_result = consumer_output[consumer][i];
      auto expected_result = ConsumerWork(kNumberOfConsumers * i + consumer);
      if (fpga_result != expected_result) {
        std::cout << "FAILED: The results are incorrect\n";
        std::cout << "On Input: " << kNumberOfConsumers * i + consumer
                  << " Expected: " << expected_result << " Got: " << fpga_result
                  << "\n";
        return 1;
      }
    }
  }

  std::cout << "PASSED: The results are correct\n";
  return 0;
}
