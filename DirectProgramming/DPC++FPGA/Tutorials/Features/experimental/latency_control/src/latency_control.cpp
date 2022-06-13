#include <CL/sycl.hpp>
#include <numeric>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

using BurstCoalescedLSU = ext::intel::experimental::lsu<
    ext::intel::experimental::burst_coalesce<true>,
    ext::intel::experimental::statically_coalesce<false>>;

int Operation(int a) { return a * 3 + 2; } // Arbitrary operations.

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class LatencyControl;

// Runs the Kernel.
void KernelRun(const std::vector<int> &in_data, std::vector<int> &out_data,
               const size_t &size) {
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif

  try {
    // Create the SYCL device queue.
    queue q(device_selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer in_buffer(in_data);
    buffer out_buffer(out_data);

    q.submit([&](handler &h) {
      accessor in_accessor(in_buffer, h, read_only);
      accessor out_accessor(out_buffer, h, write_only, no_init);

      h.single_task<LatencyControl>([=]() [[intel::kernel_args_restrict]] {
        auto in_ptr = in_accessor.get_pointer();
        auto out_ptr = out_accessor.get_pointer();

        for (size_t i = 0; i < size; i++) {
          // The following load has a label 0.
          int value = BurstCoalescedLSU::load(
              in_ptr + i, ext::oneapi::experimental::properties(
                              ext::intel::experimental::latency_anchor_id<0>));

          value = Operation(value);

          // The following store occurs exactly 5 cycles after the label-0
          // function, i.e., the load above.
          BurstCoalescedLSU::store(
              out_ptr + i, value,
              ext::oneapi::experimental::properties(
                  ext::intel::experimental::latency_constraint<
                      0, ext::intel::experimental::latency_control_type::exact,
                      5>));
        }
      });
    });
  } catch (exception const &e) {
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
}

void GoldenRun(const std::vector<int> &in_data, std::vector<int> &out_data,
               const size_t &size) {
  for (size_t i = 0; i < size; i++) {
    out_data[i] = Operation(in_data[i]);
  }
}

int main() {
  const size_t size = 5;
  std::vector<int> input_data(size);
  std::vector<int> result_kernel(size);
  std::vector<int> result_golden(size);

  // Populate in_data with incrementing values starting with 0
  std::iota(input_data.begin(), input_data.end(), 0);

  KernelRun(input_data, result_kernel, size);
  GoldenRun(input_data, result_golden, size);

  bool passed = true;

  for (int i = 0; i < size; i++) {
    if (result_kernel[i] != result_golden[i]) {
      std::cout << "Output Mismatch: \n"
                << "result_kernel[" << i << "] = " << result_kernel[i]
                << ", result_golden[" << i << "] = " << result_golden[i]
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