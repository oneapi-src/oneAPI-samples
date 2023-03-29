//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <math.h>

#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

constexpr size_t kNumCounters = 4;
constexpr int kInitialValue = 10;
constexpr unsigned kNumIncrements = 3;

namespace exp = sycl::ext::oneapi::experimental;

using IntScalar = std::array<int, 1>;
using FPGAProperties =
    decltype(exp::properties(exp::device_image_scope, exp::host_access_none));

// Array of counters that have a lifetime longer than a single kernel invocation
exp::device_global<int[kNumCounters], FPGAProperties> counters;
// Flag if the counters have been initialized - zero-initialized to `false`
exp::device_global<bool, FPGAProperties> is_counters_initialized;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class Kernel;

// Launch a kernel that increments the value of a global variable counter
// at a particular index, and returns the current value of that counter
void IncrementAndRead(sycl::queue q, IntScalar &result, int index) {
  sycl::buffer<int, 1> buffer_result(result.data(), 1);
  q.submit([&](sycl::handler &h) {
    sycl::accessor accessor_result{buffer_result, h, sycl::write_only,
                                   sycl::no_init};

    h.single_task<Kernel>([=]() [[intel::kernel_args_restrict]] {
      // Initialize counters the first time we use it
      if (!is_counters_initialized.get()) {
        for (size_t init_index = 0; init_index < kNumCounters; init_index++)
          counters[init_index] = kInitialValue;
        is_counters_initialized = true;
      }

      // Increment and read at a specific index
      counters[index]++;
      accessor_result[0] = counters[index];
    });
  });
  q.wait();
}

int main() {
  bool success = true;

  IntScalar result;

  try {
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;                  
                  

    // Increment each counter multiple times
    for (auto num_increments = 1; num_increments <= kNumIncrements;
         num_increments++) {
      // Increment each position
      for (auto counter_index = 0; counter_index < kNumCounters;
           counter_index++) {
        // Run the kernel
        IncrementAndRead(q, result, counter_index);

        // verify the results are correct
        int expected_result = kInitialValue + num_increments;
        if (result[0] != expected_result) {
          std::cerr << "device_global: mismatch at index {" << num_increments
                    << ", " << counter_index << "}: " << result[0]
                    << " != " << expected_result << " (kernel != expected)"
                    << '\n';
          success = false;
        }
      }
    }
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

  if (success) {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  }

  return 1;
}