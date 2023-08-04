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

constexpr size_t kNumIterations = 4;
constexpr unsigned kNumWeightIncrements = 3;
constexpr unsigned kVectorSize = 64;

namespace exp = sycl::ext::oneapi::experimental;

using IntScalar = std::array<int, kVectorSize>;
using FPGAProperties =
    decltype(exp::properties(exp::device_image_scope, exp::host_access_write));

// globally declared weights for the calculation
exp::device_global<int[kVectorSize], FPGAProperties> weights;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class Kernel;

// Launch a kernel that does a vector weighted add
// result = a + (weights * b)
void WeightedVectorAdd(sycl::queue q, IntScalar &a, IntScalar &b, IntScalar &result) {
  sycl::buffer<int, kVectorSize> buffer_result(result.data(), kVectorSize);
  sycl::buffer<int, kVectorSize> buffer_a(a.data(), kVectorSize);
  sycl::buffer<int, kVectorSize> buffer_b(b.data(), kVectorSize);

  q.submit([&](sycl::handler &h) {
    sycl::accessor accessor_result{buffer_result, h, sycl::write_only,
                                   sycl::no_init};
    sycl::accessor accessor_a{buffer_a, h, sycl::read_only};
    sycl::accessor accessor_b{buffer_b, h, sycl::read_only};

    h.single_task<Kernel>([=]() [[intel::kernel_args_restrict]] {
      for(auto i = 0; i < kVectorSize; i++) {
        accessor_result[i] = accessor_a[i] + (weights[i] + accessor_b[i]);
      }
    });
  });
  q.wait();
}

int main() {
  bool success = true;

  try {
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;                  
                  
    IntScalar a, b, result, host_weights;

    // Increment each counter multiple times
    for (auto weight_increment = 0; weight_increment <= kNumWeightIncrements;
         weight_increment++) {
      host_weights.fill(weight_increment);
      // Transfer data from the host to the device_global
      q.copy(host_weights, weights).wait();

      // Increment each position
      for (auto index = 0; index < kNumIterations; index++) {
        a.fill(index);
        b.fill(index);
        // Run the kernel
        WeightedVectorAdd(q, a, b, result);

        // verify the results are correct
        int expected_result = index + (weight_increment * index);
        for (auto element = 0; element < kVectorSize; element++) { 
          if (result[element] != expected_result) {
            std::cerr << "Error: for expession {" << index << " + (" 
              << weight_increment << " x " << index << ")} expected all "
              << kVectorSize 
              << "indicies to be " << expected_result << " but got " 
              << result[element] << "at index " << element << "\n";
            success = false;
          }
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