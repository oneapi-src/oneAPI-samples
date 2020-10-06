//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <vector>
#include "pipe_array.hpp"
#include "unroller.hpp"

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

constexpr size_t kNumRows = 2;
constexpr size_t kNumCols = 2;
constexpr size_t kNumberOfConsumers = kNumRows * kNumCols;
constexpr size_t kDepth = 2;

using ProducerToConsumerPipeMatrix = PipeArray<  // Defined in "pipe_array.h".
    class ProducerConsumerPipe,                  // An identifier for the pipe.
    uint64_t,  // The type of data in the pipe.
    kDepth,    // The capacity of each pipe.
    kNumRows,  // array dimension.
    kNumCols   // array dimension.
    >;

// Forward declaration of the kernel name
// (This will become unnecessary in a future compiler version.)
class ProducerTutorial;
template <size_t consumer_id> class ConsumerTutorial;

void Producer(queue &q, buffer<uint64_t, 1> &input_buffer) {
  std::cout << "Enqueuing producer...\n";

  auto e = q.submit([&](handler &h) {
    auto input_accessor = input_buffer.get_access<access::mode::read>(h);
    auto num_elements = input_buffer.get_count();
    auto num_passes = num_elements / kNumberOfConsumers;

    // The producer kernel writes to every pipe in the 2D pipe array
    h.single_task<ProducerTutorial>([=]() {
      size_t input_idx = 0;
      for (size_t pass = 0; pass < num_passes; pass++) {
        // Template-based unroll (outer "i" loop)
        Unroller<0, kNumRows>::Step([&input_idx, input_accessor](auto i) {
          // Template-based unroll (inner "j" loop)
          Unroller<0, kNumCols>::Step([&input_idx, &i, input_accessor](auto j) {
            // Write a value to the <i,j> pipe of the pipe array
            ProducerToConsumerPipeMatrix::PipeAt<i, j>::write(
                input_accessor[input_idx++]);
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
    auto output_accessor = out_buf.get_access<access::mode::discard_write>(h);
    auto num_elements = out_buf.get_count();

    // The consumer kernel reads from a single pipe, determined by consumer_id
    h.single_task<ConsumerTutorial<consumer_id>>([=]() {
      constexpr size_t consumer_x = consumer_id / kNumCols;
      constexpr size_t consumer_y = consumer_id % kNumCols;
      for (size_t i = 0; i < num_elements; ++i) {
        auto input = ProducerToConsumerPipeMatrix::PipeAt<consumer_x,
                                                          consumer_y>::read();
        uint64_t answer = ConsumerWork(input);
        output_accessor[i] = answer;
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

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif

  try {
    queue q(device_selector, dpc_common::exception_handler);

    // Enqueue producer
    buffer<uint64_t,1> producer_buffer(producer_input);
    Producer(q, producer_buffer);

    // Use verbose SYCL 1.2 syntax for the output buffer.
    // (This will become unnecessary in a future compiler version.)
    std::vector<buffer<uint64_t,1>> consumer_buffers;

    // Use template-based unroll to enqueue multiple consumers
    Unroller<0, kNumberOfConsumers>::Step([&](auto consumer_id) {
      consumer_buffers.emplace_back(consumer_output[consumer_id].data(),
                                    items_per_consumer);
      Consumer<consumer_id>(q, consumer_buffers.back());
    });

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
