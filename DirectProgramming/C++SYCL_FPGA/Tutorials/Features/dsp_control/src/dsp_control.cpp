//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"

using namespace sycl;

float subtract(float a, float b) { return a - b; }

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class GlobalControl;
class LocalControlPropagateOn;
class LocalControlPropagateOff;

// Runs the Kernel.
void KernelRun(const std::vector<float> &input_data,
               std::vector<float> &output_data_add,
               std::vector<float> &output_data_sub) {

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    // Create the SYCL device queue.
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    buffer input_buffer(input_data);
    buffer output_add_buffer(output_data_add);
    buffer output_sub_buffer(output_data_sub);

    q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_add_a(output_add_buffer, h, write_only, no_init);
      accessor output_sub_a(output_sub_buffer, h, write_only, no_init);

      // Kernel that demonstrates DSP global control.
      h.single_task<GlobalControl>([=]() [[intel::kernel_args_restrict]] {
        // Command-line option `-Xsdsp-mode=prefer-softlogic` controls both
        // addition and subtraction to be implemented in soft-logic.
        output_add_a[0] = input_a[0] + input_a[1];
        output_sub_a[0] = subtract(input_a[0], input_a[1]);
      });
    });

    q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_add_a(output_add_buffer, h, write_only, no_init);
      accessor output_sub_a(output_sub_buffer, h, write_only, no_init);

      // Kernel that demonstrates DSP local control with Propagate::On.
      h.single_task<LocalControlPropagateOn>([=
      ]() [[intel::kernel_args_restrict]] {
        // The local control library function overrides the global control.
        // Because the Propagate argument is On, not only the addition directly
        // in the lambda, but also the subtraction in the subtract() function
        // call inside the lambda are affected by the local control and will be
        // implemented in DSP.
        ext::intel::math_dsp_control<>([&] {
          output_add_a[1] = input_a[0] + input_a[1];
          output_sub_a[1] = subtract(input_a[0], input_a[1]);
        });
      });
    });

    q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_add_a(output_add_buffer, h, write_only, no_init);
      accessor output_sub_a(output_sub_buffer, h, write_only, no_init);

      // Kernel that demonstrates DSP local control with Propagate::Off.
      h.single_task<LocalControlPropagateOff>([=
      ]() [[intel::kernel_args_restrict]] {
        // The local control library function overrides the global control.
        // Because the Propagate argument is Off, only the addition directly in
        // the lambda is affected by the local control and will be implemented
        // in DSP. The subtraction in the subtract() function call is only
        // affected by the global control so will be implemented in soft-logic.
        ext::intel::math_dsp_control<ext::intel::Preference::DSP,
                                     ext::intel::Propagate::Off>([&] {
          output_add_a[2] = input_a[0] + input_a[1];
          output_sub_a[2] = subtract(input_a[0], input_a[1]);
        });
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

int main() {
  std::vector<float> input_data = {1.23f, 2.34f};
  std::vector<float> output_data_add(3);
  std::vector<float> output_data_sub(3);

  KernelRun(input_data, output_data_add, output_data_sub);

  bool passed = true;
  float golden_add = input_data[0] + input_data[1];
  float golden_sub = subtract(input_data[0], input_data[1]);

  std::string kernel_names[] = {"GlobalControl", "LocalControlPropagateOn",
                                "LocalControlPropagateOff"};
  for (int i = 0; i <= 2; i++) {
    if (output_data_add[i] != golden_add) {
      std::cout << "Kernel " << kernel_names[i] << " add output mismatch: \n"
                << "output = " << output_data_add[i]
                << ", golden = " << golden_add << "\n";
      passed = false;
    }
    if (output_data_sub[i] != golden_sub) {
      std::cout << "Kernel " << kernel_names[i] << " sub output mismatch: \n"
                << "output = " << output_data_sub[i]
                << ", golden = " << golden_sub << "\n";
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
