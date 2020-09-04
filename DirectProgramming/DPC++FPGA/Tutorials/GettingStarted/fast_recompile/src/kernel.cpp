//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

#include "kernel.hpp"

// Forward declaration of the kernel name
// (This will become unnecessary in a future compiler version.)
class VectorAdd;

void RunKernel(std::vector<float> &vec_a, std::vector<float> &vec_b,
               std::vector<float> &vec_r) {

  // Select either the FPGA emulator or FPGA device
#if defined(FPGA_EMULATOR)
  intel::fpga_emulator_selector device_selector;
#else
  intel::fpga_selector device_selector;
#endif

  try {

    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    queue q(device_selector, dpc_common::exception_handler);

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Device buffers
    buffer device_a(vec_a);
    buffer device_b(vec_b);
    // Use verbose SYCL 1.2 syntax for the output buffer.
    // (This will become unnecessary in a future compiler version.)
    buffer<float, 1> device_r(vec_r.data(), kArraySize);

    q.submit([&](handler &h) {
      // Data accessors
      auto a = device_a.get_access<access::mode::read>(h);
      auto b = device_b.get_access<access::mode::read>(h);
      auto r = device_r.get_access<access::mode::discard_write>(h);

      // Kernel executes with pipeline parallelism on the FPGA.
      // Use kernel_args_restrict to specify that a, b, and r do not alias.
      h.single_task<VectorAdd>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < kArraySize; ++i) {
          r[i] = a[i] + b[i];
        }
      });
    });

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
}
