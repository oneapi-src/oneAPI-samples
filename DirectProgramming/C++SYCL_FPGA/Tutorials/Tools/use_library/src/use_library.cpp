//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "lib.hpp"
#include "exception_handler.hpp"

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization report.
class KernelCompute;

// SYCL function to get the square a number
SYCL_EXTERNAL float SyclSquare(float x) {
  return x * x;
}

int main() {
  unsigned result = 0;

  // Select the FPGA emulator (CPU), FPGA simulator, or FPGA device
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    sycl::queue q(selector, fpga_tools::exception_handler);

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // The scalar inputs are passed to the kernel using the lambda capture,
    // but a SYCL buffer must be used to return a scalar from the kernel.
    sycl::buffer<unsigned, 1> buffer_c(&result, 1);

    //  Values used as input to the kernel
    float kA = 2.0f;
    float kB = 3.0f;


    q.submit([&](sycl::handler &h) {
      // Accessor to the scalar result
      sycl::accessor accessor_c(buffer_c, h, sycl::write_only, sycl::no_init);

      // Kernel
      h.single_task<class KernelCompute>([=]() {
        float a_sq = SyclSquare(kA);
        float b_sq = SyclSquare(kB);

        // RtlByteswap is an RTL library.
        //  - When compiled for FPGA, Verilog module byteswap_uint in lib_rtl.v
        //    is instantiated in the datapath by the compiler.
        //  - When compiled for FPGA emulator (CPU), the C model of RtlByteSwap
        //    in lib_rtl_model.cpp is used instead.
        accessor_c[0] = RtlByteswap((unsigned)(a_sq + b_sq));
      });
    });
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
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

  float kA = 2.0f;
  float kB = 3.0f;
  // Compute the expected "golden" result
  unsigned gold = (kA * kA) + (kB * kB);
  gold = gold << 16 | gold >> 16;

  // Check the results
  if (result != gold) {
    std::cout << "FAILED: result is incorrect!\n";
    return -1;
  }
  std::cout << "PASSED: result is correct!\n";
  return 0;
}

