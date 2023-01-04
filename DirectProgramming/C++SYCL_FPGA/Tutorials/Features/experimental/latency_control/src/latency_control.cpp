#include <sycl/sycl.hpp>
#include <numeric>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"

using BurstCoalescedLSU = sycl::ext::intel::experimental::lsu<
    sycl::ext::intel::experimental::burst_coalesce<true>,
    sycl::ext::intel::experimental::statically_coalesce<false>>;

int Operation(int a) { return a * 3 + 2; } // Arbitrary operations.

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class LatencyControl;

// Runs the Kernel.
void KernelRun(const std::vector<int> &in_data, std::vector<int> &out_data,
               const size_t &size) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    // Create the SYCL device queue.
    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    sycl::buffer in_buffer(in_data);
    sycl::buffer out_buffer(out_data);

    q.submit([&](sycl::handler &h) {
      sycl::accessor in_accessor(in_buffer, h, sycl::read_only);
      sycl::accessor out_accessor(out_buffer, h, sycl::write_only,
                                  sycl::no_init);

      h.single_task<LatencyControl>([=]() [[intel::kernel_args_restrict]] {
        auto in_ptr = in_accessor.get_pointer();
        auto out_ptr = out_accessor.get_pointer();

        for (size_t i = 0; i < size; i++) {
          // The following load has a label 0.
          int value = BurstCoalescedLSU::load(
              in_ptr + i,
              sycl::ext::oneapi::experimental::properties(
                  sycl::ext::intel::experimental::latency_anchor_id<0>));

          value = Operation(value);

          // The following store occurs exactly 5 cycles after the label-0
          // function, i.e., the load above.
          BurstCoalescedLSU::store(
              out_ptr + i, value,
              sycl::ext::oneapi::experimental::properties(
                  sycl::ext::intel::experimental::latency_constraint<
                      0,
                      sycl::ext::intel::experimental::latency_control_type::
                          exact,
                      5>));
        }
      });
    });
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
