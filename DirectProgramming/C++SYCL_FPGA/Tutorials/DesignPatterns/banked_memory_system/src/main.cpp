#include <iostream>

#include "exception_handler.hpp"
#include "memory_system_defines.hpp"
#include "memory_system_kernels.hpp"

// Forward declaration of kernel names.
class IDNaiveKernel;
class IDOptimizedKernel;

// Function that verifies the output from example kernels.
bool CheckOutput(sycl::queue &q) {
  SimpleOutputT result_naive;
  SimpleOutputT result_optimized;
  bool passed = true;
  for (int i = 0; i < 500; ++i) {
    result_naive = OutStreamNaiveKernel::read(q);
    result_optimized = OutStreamOptKernel::read(q);
    if (result_naive[4] != i || result_optimized[4] != i) {
      passed = false;
    }
  }
  return passed;
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
    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    // Test for simple kernels.
    q.single_task<class Input>([=]() {
      for (int i = 0; i < 500; ++i) {
        InStreamNaiveKernel::write(i);
        InStreamOptKernel::write(i);
      }
    });

    std::cout << "Launch kernel" << std::endl;

    q.single_task<IDNaiveKernel>(NaiveKernel{});
    q.single_task<IDOptimizedKernel>(OptimizedKernel{});

    std::cout << "Checking output" << std::endl;
    bool passed = CheckOutput(q);

    if (passed) {
      std::cout << "Verification PASSED.\n";
    } else {
      std::cout << "Verification FAILED.\n";
      return 1;
    }
  }
  catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }
  return 0;
}
