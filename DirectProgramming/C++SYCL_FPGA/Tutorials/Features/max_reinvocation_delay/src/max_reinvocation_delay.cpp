#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#define FACTORS 5

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class ArithmeticSequence;
class Sum;
class OutputPipe;

// Pipe between kernels
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

// Read terms from the pipe, sum them up for each sequence, write to "results"
void collect(sycl::queue &q, std::vector<int> &results, int first_term,
             int sequence_length) {
  sycl::buffer results_buffer(results);
  q.submit([&](sycl::handler &h) {
     sycl::accessor results_accessor(results_buffer, h, sycl::write_only,
                                     sycl::no_init);
     h.single_task<Sum>([=]() {
       for (int i = 0; i < FACTORS; i++) {
         int sum = 0;
         for (int j = 0; j < 10; j++) {
           sum += PipeOut::read();
         }
         results_accessor[i] = sum;
       }
     });
   }).wait();
}

// Sums up the first "sequence_length" terms for the arithmetic sequence with
// first term "first_term" and factor "factor" to compare to.
int get_arithmetic_sum(int factor, int first_term, int sequence_length) {
  return (int)((sequence_length / 2) *
               (2 * first_term + (sequence_length - 1) * factor));
}

// Verify functional correctness
void verify(std::vector<int> &results, int first_term, int sequence_length) {
  bool passed = true;
  for (int i = 0; i < FACTORS; i++) {
    int factor = i + 1;
    std::cout << std::endl
              << "Arithmetic sequence with factor = " << factor << std::endl;
    std::cout << "Sum of first " << sequence_length << " terms = " << results[i]
              << ".";
    int expected = get_arithmetic_sum(factor, first_term, sequence_length);
    bool compare = results[i] == expected;
    passed &= compare;
    if (!compare) {
      std::cout << " Error! expected " << expected << std::endl;
    } else {
      std::cout << " OK" << std::endl;
    }
  }
  std::cout << std::endl << (passed ? "PASSED" : "FAILED") << std::endl;
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

    std::vector<int> results(FACTORS);

    q.single_task<ArithmeticSequence>(
        ArithmeticSequenceKernel{first_term, sequence_length});

    collect(q, results, first_term, sequence_length);
    verify(results, first_term, sequence_length);

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