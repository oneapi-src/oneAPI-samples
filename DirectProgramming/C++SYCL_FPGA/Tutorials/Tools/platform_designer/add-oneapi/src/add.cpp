//  Copyright (c) 2023 Intel Corporation
//  SPDX-License-Identifier: MIT

#include <stdlib.h>

#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// use pipes to write into registers in the CSR address space
class OutputPipeID;

using OutputPipeProps = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::protocol<
        sycl::ext::intel::experimental::protocol_name::avalon_mm>));

using OutputPipe =
    sycl::ext::intel::experimental::pipe<OutputPipeID, int, 0, OutputPipeProps>;

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class IDAdder;

struct AdderKernel {
  int a;
  int b;

  void operator()() const {
    int sum = a + b;
    OutputPipe::write(sum);
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

    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    auto device = q.get_device();
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    int a = 3;
    int b = 76;

    int expected_sum = a + b;

    std::cout << "add two integers using CSR for input." << std::endl;

    q.single_task<IDAdder>(AdderKernel{a, b}).wait();

    // verify that outputs are correct
    passed = true;

    std::cout << "collect results." << std::endl;
    int calc_add = OutputPipe::read(q);

    std::cout << a << " + " << b << " = " << calc_add << ", expected "
              << expected_sum << ". " << std::endl;

    if (calc_add != expected_sum) {
      passed = false;
    }

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

  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
