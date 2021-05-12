/*
  Please refer to the README file for information on how and why the
  Intel(r) Dynamic Profiler for DPC++ should be used. This code sample
  does not explain the tool, rather it is an artificial example that
  demonstates the sort of code changes the profiler data can guide.
  The main content of this sample is in the README file.
*/

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <cmath>
#include <numeric>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Two identical pipes to demonstrate the behaviour before
// and after the design re-format
using ProducerToConsumerBeforePipe =
    INTEL::pipe<                           // Defined in the SYCL headers.
        class ProducerConsumerBeforePipe,  // An identifier for the pipe.
        float,                             // The type of data in the pipe.
        20>;                               // The capacity of the pipe.
using ProducerToConsumerAfterPipe =
    INTEL::pipe<class ProducerConsumerAfterPipe, float, 20>;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization report.
class ProducerBeforeKernel;
class ConsumerBeforeKernel;
class ProducerAfterKernel;
class ConsumerAfterKernel;

// kSize = # of floats to process on each kernel execution.
#if defined(FPGA_EMULATOR)
constexpr int kSize = 4096;
#else
constexpr int kSize = 262144;
#endif

// Number of iterations performed in the consumer kernels
// This controls the amount of work done by the Consumer.
constexpr int kComplexity = 2000;

// Perform two stages of processing on the input data.
// The output of ConsumerWork1 needs to go to the input
// of ConsumerWork2, so they cannot be done concurrently.
// These functions are currently doing pointless work, but
// can be replaced with more useful operations.
float ConsumerWork1(float f) {
  float output = f;
  for (int j = 0; j < kComplexity; j++) {
    output = 20 * f + j - output;
  }
  return output;
}

float ConsumerWork2(float f) {
  auto output = f;
  for (int j = 0; j < kComplexity; j++) {
    output = output + f * j;
  }
  return output;
}

/////////////////////////////////////////////////////////////
// Pre-optimization kernel versions
/////////////////////////////////////////////////////////////
// The Producer kernel reads data from a SYCL buffer and writes it to
// a pipe. This transfers the input data from the host to the Consumer kernel
// that is running concurrently.
// The Consumer kernel reads data from the pipe, performs the two ConsumerWork
// operations on the data, and writes the results to the output buffer.

void ProducerBefore(queue &q, buffer<float, 1> &buffer_a) {
  auto e = q.submit([&](handler &h) {
    // Get kernel access to the buffers
    accessor a(buffer_a, h, read_only);

    h.single_task<ProducerBeforeKernel>([=]() {
      for (int i = 0; i < kSize; i++) {
        ProducerToConsumerBeforePipe::write(a[i]);
      }
    });
  });
}

void ConsumerBefore(queue &q, buffer<float, 1> &buffer_a) {
  auto e = q.submit([&](handler &h) {
    // Get kernel access to the buffers
    accessor a(buffer_a, h, write_only, noinit);

    h.single_task<ConsumerBeforeKernel>([=]() {
      for (int i = 0; i < kSize; i++) {
        auto input = ProducerToConsumerBeforePipe::read();
        auto output = ConsumerWork1(input);
        output = ConsumerWork2(output);
        a[i] = output;
      }
    });
  });
}

/////////////////////////////////////////////////////////////
// Post-optimization kernel versions
/////////////////////////////////////////////////////////////
// The Producer kernel reads data from a SYCL buffer and does
// ConsumerWork1 on it before giving it to the concurrently
// running Consumer kernel.
// The Consumer kernel reads data from the pipe, performs the rest
// of the work (ConsumerWork2), and writes the results
// to the output buffer.

void ProducerAfter(queue &q, buffer<float, 1> &buffer_a) {
  auto e = q.submit([&](handler &h) {
    // Get kernel access to the buffers
    accessor a(buffer_a, h, read_only);

    h.single_task<ProducerAfterKernel>([=]() {
      for (int i = 0; i < kSize; i++) {
        auto input = a[i];
        auto output = ConsumerWork1(input);
        ProducerToConsumerAfterPipe::write(output);
      }
    });
  });
}

void ConsumerAfter(queue &q, buffer<float, 1> &buffer_a) {
  auto e = q.submit([&](handler &h) {
    // Get kernel access to the buffers
    accessor a(buffer_a, h, write_only, noinit);

    h.single_task<ConsumerAfterKernel>([=]() {
      for (int i = 0; i < kSize; i++) {
        auto buffer1_data = ProducerToConsumerAfterPipe::read();
        auto output = ConsumerWork2(buffer1_data);
        a[i] = output;
      }
    });
  });
}

/////////////////////////////////////////////////////////////

// Compares kernel output against expected output. Only compares part of the
// output so that this method completes quickly. This is done
// intentionally/artificially to keep host-processing time shorter than kernel
// execution time. Grabs kernel output data from its SYCL buffers.
bool ProcessOutput(buffer<float, 1> &input_buf, buffer<float, 1> &output_buf) {
  host_accessor input_buf_acc(input_buf, read_only);
  host_accessor output_buf_acc(output_buf, read_only);
  int num_errors = 0;
  int num_errors_to_print = 5;
  bool pass = true;

  // Max fractional difference between FPGA result and CPU result
  // Anything greater than this will be considered an error
  constexpr double epsilon = 0.01;

  for (int i = 0; i < kSize / 8; i++) {
    auto step1 = ConsumerWork1(input_buf_acc[i]);
    auto valid_result = ConsumerWork2(step1);

    const bool out_invalid =
        std::abs((output_buf_acc[i] - valid_result) / valid_result) > epsilon;
    if ((num_errors < num_errors_to_print) && out_invalid) {
      if (num_errors == 0) {
        pass = false;
        std::cout << "Verification failed. Showing up to "
                  << num_errors_to_print << " mismatches.\n";
      }
      std::cout << "Verification failed on the output buffer, "
                << "at element " << i << ". Expected " << valid_result
                << " but got " << output_buf_acc[i] << "\n";
      num_errors++;
    }
  }
  return pass;
}

int main() {
// Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
  std::cout << "\nThe Dynamic Profiler cannot be used in the emulator "
               "flow. Please compile to FPGA hardware to collect "
               "dynamic profiling data. \n\n";
#else
  INTEL::fpga_selector device_selector;
#endif

  try {
    queue q(device_selector, dpc_common::exception_handler);

    std::vector<float> producer_input(kSize, -1);
    std::vector<float> consumer_output_before(kSize, -1);
    std::vector<float> consumer_output_after(kSize, -1);

    // Initialize the input data
    std::iota(producer_input.begin(), producer_input.end(), 1);

    buffer producer_buffer(producer_input);
    buffer consumer_buffer_before(consumer_output_before);
    buffer consumer_buffer_after(consumer_output_after);

    std::cout << "*** Beginning execution before optimization.\n";
    ProducerBefore(q, producer_buffer);
    ConsumerBefore(q, consumer_buffer_before);
    bool pass_before = ProcessOutput(producer_buffer, consumer_buffer_before);
    if (pass_before) {
      std::cout << "Verification PASSED for run before optimization\n";
    }

    std::cout << "*** Beginning execution after optimization.\n";
    ProducerAfter(q, producer_buffer);
    ConsumerAfter(q, consumer_buffer_after);
    bool pass_after = ProcessOutput(producer_buffer, consumer_buffer_after);
    if (pass_after) {
      std::cout << "Verification PASSED for run after optimization\n";
    }

    if (pass_before && pass_after) {
      std::cout << "Verification PASSED\n";
    } else {
      std::cout << "Verification FAILED\n";
      return 1;
    }
  } catch (sycl::exception const &e) {
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
  return 0;
}
