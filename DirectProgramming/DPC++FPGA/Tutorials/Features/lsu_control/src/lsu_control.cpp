//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <numeric>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class KernelPrefetch;
class KernelBurst;
class KernelDefault;

// Aliases for LSU Control Extension types
// Implemented using template arguments such as prefetch & burst_coalesce
// on the new INTEL::lsu class to specify LSU style and modifiers
using PrefetchingLSU = INTEL::lsu<INTEL::prefetch<true>,
                                  INTEL::statically_coalesce<false>>;

using PipelinedLSU = INTEL::lsu<>;

using BurstCoalescedLSU = INTEL::lsu<INTEL::burst_coalesce<true>,
                                     INTEL::statically_coalesce<false>>;

// Input data and output data size constants
constexpr size_t kMaxVal = 128;
#if defined(FPGA_EMULATOR)
constexpr size_t kBaseVal = 1024;
#else
constexpr size_t kBaseVal = 1048576;
#endif
constexpr size_t kNum = 3;

// Return the execution time of the event, in seconds
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}

// Runs the Kernel
void KernelRun(const std::vector<int> &input_data, const size_t &input_size,
               const size_t &output_size, std::vector<int> &output_data) {
  std::fill(output_data.begin(), output_data.end(), -1);

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif
  try {
    // create the SYCL device queue
    queue q(device_selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer output_buffer(output_data);
    buffer input_buffer(input_data);

    auto e_p = q.submit([&](handler &h) {
      accessor output_a(output_buffer, h, write_only, noinit);
      accessor input_a(input_buffer, h, read_only);

      // Kernel that uses the prefetch LSU
      h.single_task<KernelPrefetch>([=]() [[intel::kernel_args_restrict]] {
        auto input_ptr = input_a.get_pointer();
        auto output_ptr = output_a.get_pointer();

        int total = 0;
        for (size_t i = 0; i < input_size; i++) {
          total += PrefetchingLSU::load(input_ptr + i);
        }
        output_ptr[0] = total;
      });
    });

    auto e_b = q.submit([&](handler &h) {
      accessor output_a(output_buffer, h, write_only, noinit);
      accessor input_a(input_buffer, h, read_only);
      
      // Kernel that uses the burst-coalesced LSU
      h.single_task<KernelBurst>([=]() [[intel::kernel_args_restrict]] {
        auto input_ptr = input_a.get_pointer();
        auto output_ptr = output_a.get_pointer();

        int total = 0;
        for (size_t i = 0; i < input_size; i++) {
          total += BurstCoalescedLSU::load(input_ptr + i);
        }
        output_ptr[1] = total;
      });
    });

    auto e_d = q.submit([&](handler &h) {
      accessor output_a(output_buffer, h, write_only, noinit);
      accessor input_a(input_buffer, h, read_only);
      
      // Kernel that uses the default LSUs
      h.single_task<KernelDefault>([=]() [[intel::kernel_args_restrict]] {
        auto input_ptr = input_a.get_pointer();
        auto output_ptr = output_a.get_pointer();

        int total = 0;
        for (size_t i = 0; i < input_size; i++) {
          total += input_ptr[i];
        }
        output_ptr[2] = total;
      });
    });

    // Measure the execution time of each kernel 
    double p_time = GetExecutionTime(e_p);
    double b_time = GetExecutionTime(e_b);
    double d_time = GetExecutionTime(e_d);
    double input_size_mb = (input_size * sizeof(int)/(1024*1024));
    std::cout << "Kernel throughput with prefetch LSU: " 
              << (input_size_mb/p_time) << " MB/s \n";
    std::cout << "Kernel throughput with burst-coalesced LSU: " 
              << (input_size_mb/b_time) << " MB/s \n";
    std::cout << "Kernel throughput without LSU controls: " 
              << (input_size_mb/d_time) << " MB/s \n";

  } catch (exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
}

// This host side function performs the same computation as the device side
// kernel, and is used to verify functional correctness.
void GoldenRun(const std::vector<int> &input_data, const size_t &input_size,
               const size_t &output_size, std::vector<int> &output_data) {
  std::fill(output_data.begin(), output_data.end(), -1);

  for (size_t i = 0; i < output_size; i++) {
    int total = 0;
    for (size_t j = 0; j < input_size; j++) {
      // Match formulas from kernel above
      total += input_data[j];
    }
    output_data[i] = total;
  }
}

int main() {
  bool passed = true;
  const size_t input_size = kBaseVal + rand() % kMaxVal;
  const size_t output_size = kNum;

  std::vector<int> input_data(input_size);
  std::vector<int> result_golden(output_size);
  std::vector<int> result_kernel(output_size);

  // Populate input_data with incrementing values starting with 0
  std::iota(input_data.begin(), input_data.end(), 0);

  GoldenRun(input_data, input_size, output_size, result_golden);
  KernelRun(input_data, input_size, output_size, result_kernel);

  for (size_t i = 0; i < output_size; i++) {
    if (result_kernel[i] != result_golden[i]) {
      std::cout << "Output Mismatch: \n"
                << "result_kernel[" << i << "] vs result_golden [" << i
                << "] = " << result_kernel[i] << "," << result_golden[i]
                << " \n";
      passed = false;
    }
  }

  if (passed) {
    std::cout << "PASSED: all kernel results are correct.\n";
  } else {
    std::cout << "FAILED\n";
  }
  return passed ? 0 : 1;
}
