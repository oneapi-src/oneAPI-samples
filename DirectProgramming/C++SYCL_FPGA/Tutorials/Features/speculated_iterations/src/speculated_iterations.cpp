//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include "exception_handler.hpp"

// Use smaller values if run on the emulator or simulator to keep the CPU
// runtime/simulation time reasonable
// Use the largest possible int values on the FPGA to show the difference in
// performance with and without speculated_iterations
#if defined(FPGA_EMULATOR)
constexpr float kUpper = 3.0f;
constexpr size_t kExpectedIterations = 1e3;
#elif defined(FPGA_SIMULATOR)
constexpr float kUpper = 2.0f;
constexpr size_t kExpectedIterations = 1e2;
#else
constexpr float kUpper = 8.0f;
constexpr size_t kExpectedIterations = 1e8;
#endif

using namespace sycl;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int N> class KernelCompute;

template <int spec_iter, bool first_call = false>
void ComplexExit(float bound, int &res) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  double kernel_time_ms = 0.0;
  try {
    // create the device queue with profiling enabled
    auto prop_list = property_list{property::queue::enable_profiling()};
    queue q(selector, fpga_tools::exception_handler, prop_list);

    if constexpr (first_call){
      auto device = q.get_device();

      std::cout << "Running on device: "
                << device.get_info<sycl::info::device::name>().c_str()
                << std::endl;
    }

    // The scalar inputs are passed to the kernel using the lambda capture,
    // but a SYCL buffer must be used to return a scalar from the kernel.
    buffer<int, 1> buffer_res(&res, 1);

    event e = q.submit([&](handler &h) {
      accessor accessor_res(buffer_res, h, write_only, no_init);

      h.single_task<class KernelCompute<spec_iter>>([=]() {
        int x = 1;

        // Computing the exit condition of this loop is a complex operation.
        // Since the value of var is not known at compile time, the loop
        // trip count is variable and the exit condition must be evaluated at
        // each iteration.
        [[intel::speculated_iterations(spec_iter)]]
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
    std::cerr << "Caught synchronous SYCL exception:\n" << exc.what() << "\n";
    if (exc.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
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

  float bound = kUpper;

  // We don't want "bound" to be a compile-time known constant value
  if (argc > 1) {
    std::string option(argv[1]);
    bound = std::stoi(option);
  }

  // result variables
  int r0, r1, r2;

// Choose the number of speculated iterations based on the FPGA board selected.
// This reflects compute latency differences on different hardware
// architectures, and is a low-level optimization.
#if defined(A10)
  ComplexExit<0, true>(bound, r0);
  ComplexExit<10>(bound, r1);
  ComplexExit<27>(bound, r2);
#elif defined(S10)
  ComplexExit<0, true>(bound, r0);
  ComplexExit<10>(bound, r1);
  ComplexExit<54>(bound, r2);
#elif defined(Agilex7)
  ComplexExit<0, true>(bound, r0);
  ComplexExit<10>(bound, r1);
  ComplexExit<50>(bound, r2);
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
