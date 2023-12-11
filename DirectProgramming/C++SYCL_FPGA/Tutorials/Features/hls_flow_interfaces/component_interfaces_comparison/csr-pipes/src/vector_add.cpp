#include <iostream>

// oneAPI headers
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "exception_handler.hpp"

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class SimpleVAddPipes;

// Forward declare pipe names to reduce name mangling
class IDPipeA;
class IDPipeB;
class IDPipeC;

constexpr int kVectorSize = 256;

using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>));

using InputPipeA =
    sycl::ext::intel::experimental::pipe<IDPipeA, int, 0,
                                         PipeProps>;
using InputPipeB =
    sycl::ext::intel::experimental::pipe<IDPipeB, int, 0,
                                         PipeProps>;

using CSRPipeProps = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::protocol_avalon_mm_uses_ready));

// this csr pipe will only be read from and written to once
using OutputPipeC =
    sycl::ext::intel::experimental::pipe<IDPipeC, int, 0,
                                         CSRPipeProps>;

struct SimpleVAddKernelPipes {
  int len;

  void operator()() const {
    int sum_total = 0;
    for (int idx = 0; idx < len; idx++) {
      int a_val = InputPipeA::read();
      int b_val = InputPipeB::read();
      int sum = a_val + b_val;

      sum_total += sum;
    }

    // Write to OutputPipeC only once per kernel invocation
    OutputPipeC::write(sum_total);
  }
};

int main() {
  try {
    // Use compile-time macros to select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    //  - the simulator device
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    int count = kVectorSize;  // pass array size by value

    int expected_sum = 0;

    // push data into pipes
    int *a = new int[count];
    int *b = new int[count];
    for (int i = 0; i < count; i++) {
      a[i] = i;
      b[i] = (count - i);

      expected_sum += (a[i] + b[i]);
      // When writing to a host pipe in non kernel code, 
      // you must pass the sycl::queue as the first argument
      InputPipeA::write(q, a[i]);
      InputPipeB::write(q, b[i]);
    }

    std::cout << "Add two vectors of size " << count << std::endl;

    q.single_task<SimpleVAddPipes>(SimpleVAddKernelPipes{count});

    // verify that outputs are correct
    bool passed = true;

    // only need to read from OutputPipeC once, since the kernel only wrote to it once
    int calc = OutputPipeC::read(q);
    if (calc != expected_sum) {
      std::cout << "result " << calc << ", expected (" << expected_sum << ")"
                << std::endl;
      passed = false;
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    delete[] a;
    delete[] b;

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::terminate();
  }
}