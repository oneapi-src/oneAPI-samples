#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
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

// Minimum capacity of a pipe.
// Set to 0 to allow the compiler to save area if possible.
constexpr size_t kPipeMinCapacity = 0;

// Pipes
class IDInputPipeA;
class IDInputPipeB;
class IDInputPipeC;
class IDOutputPipeZ;

using InputPipeA = 
  sycl::ext::intel::experimental::pipe<IDInputPipeA, float, kPipeMinCapacity>;
using InputPipeB = 
  sycl::ext::intel::experimental::pipe<IDInputPipeB, float, kPipeMinCapacity>;
using InputPipeC = 
  sycl::ext::intel::experimental::pipe<IDInputPipeC, float, kPipeMinCapacity>;
using OutputPipeZ = 
  sycl::ext::intel::experimental::pipe<IDOutputPipeZ, float, kPipeMinCapacity>;

float OpSqrt(D3Vector val, const D3Vector coef) {
  float res = sqrt(val.d[0] * coef.d[0] + val.d[1] * coef.d[1] + val.d[2] * coef.d[2]);
  return res;
}

struct VectorOp {
  int len;

  void operator()() const {
    constexpr D3Vector coef1 = {0.2, 0.3, 0.4};
    constexpr D3Vector coef2 = {0.6, 0.7, 0.8};

    D3Vector new_item;
    
    for (int i = 0; i < len; i++) {
      D3Vector item;
	    item.d[0] = InputPipeA::read();
	    item.d[1] = InputPipeB::read();
	    item.d[2] = InputPipeC::read();
      new_item.d[i] = OpSqrt(item, coef1);
    }
	
	OutputPipeZ::write(OpSqrt(new_item, coef2));
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

    // input data
    for (int i = 0; i < kVectSize; i++) {
      InputPipeA::write(q, test_vecs[i][0]);
      InputPipeB::write(q, test_vecs[i][1]);
      InputPipeC::write(q, test_vecs[i][2]);
    }

    std::cout << "Processing vector of size " << kVectSize << std::endl;

    float result[N];
	  sycl::event e;
    for (int i = 0; i < N; i++) {
      std::cout << "Calling kernel " << i << std::endl;
      e = q.single_task<VectorOpID>(VectorOp{kVectSize});
    }

    // verify that result is correct
    passed = true;
    for (int i = 0; i < N; i++) {
      std::cout << "Reading result " << i << std::endl;
      result[i] = OutputPipeZ::read(q);
	  }
    for (int i = 1; i < N; i++) {
      if (result[i] != result[i-1]) {
        std::cout << "idx=" << i << ", naive result " << result[i] << ", previously " << result[i-1] << std::endl;
        passed = false;
	    }
    }
    // Wait for kernel to exit
    e.wait();
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;


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