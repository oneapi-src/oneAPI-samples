#include <iostream>

#include "exception_handler.hpp"
#include "memory_system_defines.hpp"
#include "memory_system_kernels.hpp"

// Forward declaration of kernel names.
class ID_NaiveKernel;
class ID_OptimizedKernel;

// Function that verifies the output from example kernels.
bool CheckOutput(sycl::queue &q) {
  SimpleOutputT resultNaive;
  SimpleOutputT resultOptimized;
  bool passed = true;
  for (int i = 0; i < 500; ++i) {
    resultNaive = OutStream_NaiveKernel::read(q);
    resultOptimized = OutStream_OptKernel::read(q);
    if (resultNaive[4] != i || resultOptimized[4] != i) {
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
        InStream_NaiveKernel::write(i);
        InStream_OptKernel::write(i);
      }
    });

    std::cout << "Launch kernel" << std::endl;

    q.single_task<ID_NaiveKernel>(NaiveKernel{});
    q.single_task<ID_OptimizedKernel>(OptimizedKernel{});

    std::cout << "Checking output" << std::endl;
    bool passed = CheckOutput(q);

    if (passed) {
      std::cout << "Verification PASSED.\n";
    } else {
      std::cout << "Verification FAILED.\n";
    }
  }
  catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }
}
