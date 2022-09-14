//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <math.h>

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

constexpr size_t kNumCounters = 4;
constexpr int kInitialValue = 10;
constexpr unsigned kNumIncrements = 3;

using IntScalar = std::array<int, 1>;
using FPGAProperties = decltype(properties(device_image_scope));

// Array of counters that have a lifetime longer than a single kernel invocation
device_global<int[kNumCounters], FPGAProperties> counters;
// Flag if the counters have been initialized
device_global<bool, FPGAProperties> is_counters_initialized;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class Kernel;

// Launch a kernel that increments the value of a global variable counter
// at a particular index, and returns the current value of that counter
void IncrementAndRead(const device_selector &selector, IntScalar &result,
                      int index) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer<int, 1> buffer_result(result.data(), 1);
    q.submit([&](handler &h) {
      accessor accessor_result{buffer_result, h, write_only, no_init};

      h.single_task<Kernel>([=]() [[intel::kernel_args_restrict]] {
        // Initialize counters the first time we use it
        if (!is_counters_initialized.get()) {
          for (size_t index = 0; index < kNumCounters; index++)
            counters[index] = kInitialValue;
          is_counters_initialized = true;
        }

        // Increment and read at a specific index
        counters[index]++;
        accessor_result[0] = counters[index];
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
}

int main() {
  bool success = true;

  IntScalar result;

#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  // Increment each counter multiple times
  for (auto num_increments = 1; num_increments <= kNumIncrements; num_increments++) {
    // Increment each position
    for (auto counter_index = 0; counter_index < kNumCounters;
         counter_index++) {
      // Run the kernel
      IncrementAndRead(selector, result, counter_index);

      // verify the results are correct
      int expected_result = kInitialValue + num_increments;
      if (result[0] != expected_result) {
        std::cout << "device_global: mismatch at index " << counter_index
                  << ": " << result[0] << " != " << expected_result
                  << " (kernel != expected)" << '\n';
        success = false;
      }
    }
  }

  if (success) {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  }

  return 1;
}
