#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class VectorAddID;

class VectorAdd {
 public:
  int *a_in;
  int *b_in;
  int *c_out;
  int len;

  void operator()() const {
    for (int idx = 0; idx < len; idx++) {
      int a_val = a_in[idx];
      int b_val = b_in[idx];
      int sum = a_val + b_val;
      c_out[idx] = sum;
    }
  }
};

#define VECT_SIZE 256

int main() {
  bool passed = false;

  try {
// This design is tested with 2023.0, but also accounts for a syntax change in
// 2023.1
#if __INTEL_CLANG_COMPILER >= 20230100
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
#elif __INTEL_CLANG_COMPILER >= 20230000
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector{};
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector{};
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector{};
#endif
#else
    assert(false) && "this design requires oneAPI 2023.0 or 2023.1!"
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    int count = VECT_SIZE;  // pass array size by value

    // declare arrays and fill them
    // allocate in shared memory so the kernel can see them
    int *a = sycl::malloc_shared<int>(count, q);
    int *b = sycl::malloc_shared<int>(count, q);
    int *c = sycl::malloc_shared<int>(count, q);
    for (int i = 0; i < count; i++) {
      a[i] = i;
      b[i] = (count - i);
    }

    std::cout << "add two vectors of size " << count << std::endl;

    q.single_task<VectorAddID>(VectorAdd{a, b, c, count}).wait();

    // verify that VC is correct
    passed = true;
    for (int i = 0; i < count; i++) {
      int expected = a[i] + b[i];
      if (c[i] != expected) {
        std::cout << "idx=" << i << ": result " << c[i] << ", expected ("
                  << expected << ") A=" << a[i] << " + B=" << b[i] << std::endl;
        passed = false;
      }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code.
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

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}