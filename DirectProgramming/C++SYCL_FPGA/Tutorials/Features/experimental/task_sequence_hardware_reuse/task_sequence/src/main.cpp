#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/experimental/task_sequence.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class VectorOpID;

constexpr int kVectSize = 3;
constexpr int N = 5;
struct D3Vector {
  float d[3];
};

// The square-root of a dot-product is an expensive operation.
float OpSqrt(D3Vector val, const D3Vector coef) {
  float res = sqrt(val.d[0] * coef.d[0] + val.d[1] * coef.d[1] + val.d[2] * coef.d[2]);
  return res;
}

struct VectorOp {
  D3Vector *a_in;
  float *z_out;
  int len;

  void operator()() const {
    constexpr D3Vector coef1 = {0.2, 0.3, 0.4};
    constexpr D3Vector coef2 = {0.6, 0.7, 0.8};

    D3Vector new_item;
    
    // Object declarations of a parameterized task_sequence class must be local, which means global declarations and dynamic allocations are not allowed.
    // Declare the task sequence object outside the for loop so that the hardware can be shared at the return point.
    sycl::ext::intel::experimental::task_sequence<OpSqrt> task_a;
    
    for (int i =0; i < len; i++) {
      task_a.async(a_in[i], coef1);
    }

    for (int i =0; i < len; i++) {
      new_item.d[i] = task_a.get();
    }

    task_a.async(new_item, coef2);
    z_out[0] = task_a.get();
  }
};

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

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // initialize input D3Vector
    constexpr float test_vecs[kVectSize][3] = {
      {.49, .26, .82},
      {.78, .43, .92},
      {.17, .72, .34}
    };

    // Create USM shared allocations in the specified buffer_location. 
    // You can also use host allocations with malloc_host(...) API
    D3Vector *a = sycl::malloc_shared<D3Vector>(kVectSize, q);
    float *z = sycl::malloc_shared<float>(1, q);
    for (int i = 0; i < kVectSize; i++) {
      a[i].d[0] = test_vecs[i][0];
      a[i].d[1] = test_vecs[i][1];
      a[i].d[2] = test_vecs[i][2];
    }

    std::cout << "Processing vector of size " << kVectSize << std::endl;

    float result[N];
    for (int i = 0; i < N; i++) {
      q.single_task<VectorOpID>(VectorOp{a, z, kVectSize}).wait();
	    result[i] = z[0];
    }

    // verify that result is correct
    passed = true;
    for (int i = 1; i < N; i++) {
      if (result[i] != result[i-1]) {
        std::cout << "idx=" << i << ", async result " << result[i] << ", previously " << result[i-1] << std::endl;
        passed = false;
	    }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    sycl::free(a, q);
    sycl::free(z, q);

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