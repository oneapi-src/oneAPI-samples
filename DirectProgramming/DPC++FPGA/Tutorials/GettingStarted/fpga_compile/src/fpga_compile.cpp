//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// Vector size for this example
constexpr size_t kSize = 1024;

// Forward declaration of the kernel name
// (This will become unnecessary in a future compiler version.)
class VectorAdd;


int main() {

  // Set up three vectors and fill two with random values.
  std::vector<int> vec_a(kSize), vec_b(kSize), vec_r(kSize);
  for (int i = 0; i < kSize; i++) {
    vec_a[i] = rand();
    vec_b[i] = rand();
  }

  // Select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif

  try {

    // Create a queue bound to the chosen device.
    // If the device is unavailable, a SYCL runtime exception is thrown.
    queue q(device_selector, dpc_common::exception_handler);

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    {
      // Create buffers to share data between host and device.
      // The runtime will copy the necessary data to the FPGA device memory
      // when the kernel is launched.
      buffer buf_a(vec_a);
      buffer buf_b(vec_b);
      // Use verbose SYCL 1.2 syntax for the output buffer.
      // (This will become unnecessary in a future compiler version.)
      buffer<int,1> buf_r(vec_r.data(), kSize);


      // Submit a command group to the device queue.
      q.submit([&](handler& h) {

        // The SYCL runtime uses the accessors to infer data dependencies.
        // A "read" accessor must wait for data to be copied to the device
        // before the kernel can start. A "write discard" accessor does not.
        auto a = buf_a.get_access<access::mode::read>(h);
        auto b = buf_b.get_access<access::mode::read>(h);
        auto r = buf_r.get_access<access::mode::discard_write>(h);

        // The kernel uses single_task rather than parallel_for.
        // The task's for loop is executed in pipeline parallel on the FPGA,
        // exploiting the same parallelism as an equivalent parallel_for.
        h.single_task<VectorAdd>([=]() {
          for (int i = 0; i < kSize; ++i) {
            r[i] = a[i] + b[i];
          }
        });
      });

      // The buffer destructor is invoked when the buffers pass out of scope.
      // buf_r's destructor updates the content of vec_r on the host.
    }

    // The queue destructor is invoked when q passes out of scope.
    // q's destructor invokes q's exception handler on any device exceptions.
  }
  catch (sycl::exception const& e) {
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

  // Check the results.
  int correct = 0;
  for (int i = 0; i < kSize; i++) {
    if ( vec_r[i] == vec_a[i] + vec_b[i] ) {
      correct++;
    }
  }

  // Summarize and return.
  if (correct == kSize) {
    std::cout << "PASSED: results are correct\n";
  } else {
    std::cout << "FAILED: results are incorrect\n";
  }

  return !(correct == kSize);
}
