#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class VectorAddID;

struct VectorAdd {
  int *const a_in;
  int *const b_in;
  int *const c_out;
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

constexpr int kVectSize = 256;

int main() {
  bool passed = false;

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

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    // make sure the device supports USM host allocations
    if (!device.has(sycl::aspect::usm_host_allocations)) {
      std::cerr << "This design must either target a board that supports USM "
                   "Host/Shared allocations, or IP Component Authoring. "
                << std::endl;
      std::terminate();
    }

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // declare arrays and fill them
    // allocate in shared memory so the kernel can see them
    int *a = sycl::malloc_shared<int>(kVectSize, q);
    int *b = sycl::malloc_shared<int>(kVectSize, q);
    int *c = sycl::malloc_shared<int>(kVectSize, q);
    for (int i = 0; i < kVectSize; i++) {
      a[i] = i;
      b[i] = (kVectSize - i);
    }

    std::cout << "add two vectors of size " << kVectSize << std::endl;

    q.single_task<VectorAddID>(VectorAdd{a, b, c, kVectSize}).wait();

    // verify that VC is correct
    passed = true;
    for (int i = 0; i < kVectSize; i++) {
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