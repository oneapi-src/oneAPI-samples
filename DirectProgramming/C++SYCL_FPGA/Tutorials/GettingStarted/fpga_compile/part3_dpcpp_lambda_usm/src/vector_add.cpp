#include <iostream>
#include <vector>

// oneAPI headers
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class VectorAddID;

void VectorAdd(const int *vec_a_in, const int *vec_b_in, int *vec_c_out,
               int len) {
  for (int idx = 0; idx < len; idx++) {
    int a_val = vec_a_in[idx];
    int b_val = vec_b_in[idx];
    int sum = a_val + b_val;
    vec_c_out[idx] = sum;
  }
}

constexpr int kVectSize = 256;

int main() {
  bool passed = true;
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
    sycl::queue q(selector);

    // make sure the device supports USM host allocations
    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // declare arrays and fill them
    // allocate in shared memory so the kernel can see them
    int *vec_a = malloc_shared<int>(kVectSize, q);
    int *vec_b = malloc_shared<int>(kVectSize, q);
    int *vec_c = malloc_shared<int>(kVectSize, q);
    for (int i = 0; i < kVectSize; i++) {
      vec_a[i] = i;
      vec_b[i] = (kVectSize - i);
    }

    std::cout << "add two vectors of size " << kVectSize << std::endl;

    q.single_task<VectorAddID>([=]() { 
        VectorAdd(vec_a, vec_b, vec_c, kVectSize); 
    })
    .wait();

    // verify that vec_c is correct
    for (int i = 0; i < kVectSize; i++) {
      int expected = vec_a[i] + vec_b[i];
      if (vec_c[i] != expected) {
        std::cout << "idx=" << i << ": result " << vec_c[i] << ", expected ("
                  << expected << ") A=" << vec_a[i] << " + B=" << vec_b[i]
                  << std::endl;
        passed = false;
      }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    free(vec_a, q);
    free(vec_b, q);
    free(vec_c, q);
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