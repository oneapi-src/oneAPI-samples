//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <array>
#include <iomanip>
#include <iostream>
#include <type_traits>

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

// Use smaller values if run on the emulator to keep the CPU runtime reasonable
// Use the largest possible int values on the FPGA to show the difference in
// performance with and without speculated_iterations
#if defined(FPGA_EMULATOR)
constexpr float kUpper = 3.0f;
constexpr size_t kExpectedIterations = 1e3;
#else
constexpr float kUpper = 8.0f;
constexpr size_t kExpectedIterations = 1e8;
#endif

using namespace sycl;

// This is the class used to name the kernel for the runtime.
// This must be done when the kernel is expressed as a lambda.
template <int N> class KernelCompute;

template <int spec_iter>
void ComplexExit(const device_selector &selector, float bound, int &res) {
  double kernel_time_ms = 0.0;
  try {
    // create the device queue with profiling enabled
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(selector, dpc_common::exception_handler, prop_list);

    // The scalar inputs are passed to the kernel using the lambda capture,
    // but a SYCL buffer must be used to return a scalar from the kernel.
    buffer<int, 1> buffer_res(&res, 1);

    event e = q.submit([&](handler &h) {
      auto accessor_res = buffer_res.get_access<access::mode::discard_write>(h);

      h.single_task<class KernelCompute<spec_iter>>([=]() {
        int x = 1;

        // Computing the exit condition of this loop is a complex operation.
        // Since the value of var is not known at compile time, the loop
        // trip count is variable and the exit condition must be evaluated at
        // each iteration.
        [[intelfpga::speculated_iterations(spec_iter)]]
        while (sycl::log10((float)(x)) < bound) {
          x++;
        }

        accessor_res[0] = x;
      });
    });

    // get the kernel time in milliseconds
    // this excludes memory transfer and queuing overhead
    double startk =
        e.template get_profiling_info<info::event_profiling::command_start>();
    double endk =
        e.template get_profiling_info<info::event_profiling::command_end>();
    kernel_time_ms = (endk - startk) * 1e-6;

  } catch (exception const &exc) {
    std::cout << "Caught synchronous SYCL exception:\n" << exc.what() << "\n";
    if (exc.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  // MFLOPs = mega floating point operations per second
  double mflops = (double)(kExpectedIterations) / kernel_time_ms;

  std::cout << "Speculated Iterations: " << spec_iter
            << " -- kernel time: " << kernel_time_ms << " ms\n";

  std::cout << std::fixed << std::setprecision(0)
            << "Performance for kernel with " << spec_iter
            << " speculated iterations: " << mflops << " MFLOPs\n";
}

int main(int argc, char *argv[]) {
#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  float bound = kUpper;

  // We don't want "bound" to be a compile-time known constant value
  if (argc > 1) {
    std::string option(argv[1]);
    bound = std::stoi(option);
  }

  // result variables
  int r0, r1, r2;

// Choose the number of speculated iterations based on the FPGA board selected.
// This reflects compute latency differences on different hardware architectures,
// and is a low-level optimization.
#if defined(A10)
  ComplexExit<0>(selector, bound, r0);
  ComplexExit<10>(selector, bound, r1);
  ComplexExit<27>(selector, bound, r2);
#elif defined(S10)
  ComplexExit<0>(selector, bound, r0);
  ComplexExit<10>(selector, bound, r1);
  ComplexExit<54>(selector, bound, r2);
#else
  std::static_assert(false, "Invalid FPGA board macro");
#endif

  bool passed = true;

  if (std::fabs(std::log10(r0) - bound) > 1e-5) {
    std::cout << "Test 0 result mismatch " << std::log10(r0)
              << " not within 0.00001 of " << bound << "\n";
    passed = false;
  }

  if (std::fabs(std::log10(r1) - bound) > 1e-5) {
    std::cout << "Test 1 result mismatch " << std::log10(r1)
              << " not within 0.00001 of " << bound << "\n";
    passed = false;
  }

  if (std::fabs(std::log10(r2) - bound) > 1e-5) {
    std::cout << "Test 2 result mismatch " << std::log10(r2)
              << " not within 0.00001 of " << bound << "\n";
    passed = false;
  }


  std::cout << (passed ? "PASSED: The results are correct" : "FAILED") << "\n";

  return passed ? 0 : -1;
}

