#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

constexpr int kFactors = 5;

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class ArithmeticSequence;
class ResultsPipe;

// Results pipe from device back to host
using PipeResults = sycl::ext::intel::experimental::pipe<ResultsPipe, int>;

// Computes and outputs the first "sequence_length" terms of the arithmetic
// sequences with first term "first_term" and factors 1 through kFactors.
struct ArithmeticSequenceKernel {
  int first_term;
  int sequence_length;

  void operator()() const {
    for (int factor = 1; factor <= kFactors; factor++) {
      [[intel::max_reinvocation_delay(1)]] // NO-FORMAT: Attribute
      for (int i = 0; i < sequence_length; i++) {
        PipeResults::write(first_term + i * factor);
      }
    }
  }
};

int main() {

  try {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    sycl::queue q(selector, fpga_tools::exception_handler);
    auto device = q.get_device();
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    int first_term = 0;
    int sequence_length = 10;

    q.single_task<ArithmeticSequence>(
        ArithmeticSequenceKernel{first_term, sequence_length});

    // Verify functional correctness
    bool passed = true;
    for (int factor = 1; factor <= kFactors; factor++) {
      std::cout << "Calculating arithmetic sequence with factor = " << factor
                << std::endl;
      for (int i = 0; i < sequence_length; i++) {
        int val_device = PipeResults::read(q);
        int val_host = first_term + i * factor;
        passed &= (val_device == val_host);
        if (val_device != val_host) {
          std::cout << "Error: expected " << val_host << ", got " << val_device
                    << std::endl;
        }
      }
    }
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    
  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cerr << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;
    std::terminate();
  }
}