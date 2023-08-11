#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#define FACTORS 5

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class ArithmeticSequence;
class Sum;
class OutputPipe;

// Inter-kernel pipe
using PipeOut = sycl::ext::intel::pipe<OutputPipe, int, 50>;

// Computes and outputs the first "sequence_length" terms of the arithmetic
// sequences with first term "first_term" and factors 1 through FACTORS.
struct ArithmeticSequenceKernel {
  int first_term;
  int sequence_length;

  void operator()() const {
    for (int i = 0; i < FACTORS; i++) {
      int factor = i + 1;
      [[intel::max_reinvocation_delay(1)]] // NO-FORMAT: Attribute
      for (int j = 0; j < sequence_length; j++) {
        PipeOut::write(first_term + j * factor);
      }
    }
  }
};

// Sums up the first "sequence_length" terms for each sequence and writes the
// results to global memory.
struct SumKernel {
  int sequence_length;
  int *result;

  void operator()() const {
    for (int i = 0; i < FACTORS; i++) {
      int sum = 0;
      for (int j = 0; j < sequence_length; j++) {
        sum += PipeOut::read();
      }
      result[i] = sum;
    }
  }
};

// Sums up the first "sequence_length" terms for the arithmetic sequence with
// first term "first_term" and factor "factor" to compare to.
int get_arithmetic_sum(int first_term, int factor, int sequence_length) {
  return (int)((sequence_length / 2) *
               (2 * first_term + (sequence_length - 1) * factor));
}

int main() {

  try {
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    sycl::queue q(selector);

    auto device = q.get_device();
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    int first_term = 0;
    int sequence_length = 10;

    int *result = sycl::malloc_shared<int>(FACTORS, q);

    q.single_task<ArithmeticSequence>(
        ArithmeticSequenceKernel{first_term, sequence_length});
    q.single_task<Sum>(SumKernel{sequence_length, result}).wait();

    // Verify results
    bool passed = true;
    for (int i = 0; i < FACTORS; i++) {
      int factor = i + 1;
      std::cout << std::endl
                << "Arithmetic sequence with factor = " << factor << std::endl;
      std::cout << "Sum of first " << sequence_length
                << " terms = " << result[i] << ".";
      int expected = get_arithmetic_sum(first_term, factor, sequence_length);
      bool compare = result[i] == expected;
      passed &= compare;
      if (!compare) {
        std::cout << " Error! expected " << expected << std::endl;
      } else {
        std::cout << " OK" << std::endl;
      }
    }
    std::cout << std::endl << (passed ? "PASSED" : "FAILED") << std::endl;
    sycl::free(result, q);

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