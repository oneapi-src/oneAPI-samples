#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

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
    int *vec_a = new int[kVectSize];
    int *vec_b = new int[kVectSize];
    int *vec_c = new int[kVectSize];
    for (int i = 0; i < kVectSize; i++) {
      vec_a[i] = i;
      vec_b[i] = (kVectSize - i);
    }

    std::cout << "add two vectors of size " << kVectSize << std::endl;
    {
      // copy the input arrays to buffers to share with kernel
      sycl::buffer buffer_a{vec_a, sycl::range(kVectSize)};
      sycl::buffer buffer_b{vec_b, sycl::range(kVectSize)};
      sycl::buffer buffer_c{vec_c, sycl::range(kVectSize)};

      q.submit([&](sycl::handler &h) {
        // use accessors to interact with buffers from device code
        sycl::accessor accessor_a{buffer_a, h, sycl::read_only};
        sycl::accessor accessor_b{buffer_b, h, sycl::read_only};
        sycl::accessor accessor_c{buffer_c, h, sycl::read_write, sycl::no_init};

        h.single_task<VectorAddID>([=]() {
          VectorAdd(&accessor_a[0], &accessor_b[0], &accessor_c[0], kVectSize);
        });
      });
    }
    // result is copied back to host automatically when accessors go out of
    // scope.

    // verify that VC is correct
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

    delete[] vec_a;
    delete[] vec_b;
    delete[] vec_c;
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