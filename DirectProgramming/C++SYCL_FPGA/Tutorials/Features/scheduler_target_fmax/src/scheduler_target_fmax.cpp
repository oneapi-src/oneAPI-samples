//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"

using namespace sycl;

constexpr unsigned kSeed = 1313;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class Default;
class Fmax480;
class Fmax240;
class Fmax240II;

// Runs the Kernel
void KernelRun(size_t size, const std::vector<char> &input_data,
               std::vector<unsigned> &output_data) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    // create the SYCL device queue
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    buffer input_buffer(input_data);
    buffer output_buffer(output_data);

    q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_a(output_buffer, h, write_only, no_init);

      h.single_task<Default>([=]() [[intel::kernel_args_restrict]] {
        unsigned hash = 0;
        for (size_t i = 0; i < size; i++) {
          hash = (hash * kSeed) + input_a[i];
        }
        output_a[0] = hash;
      });
    });

    q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_a(output_buffer, h, write_only, no_init);

      h.single_task<Fmax480>([=]() [[intel::kernel_args_restrict,
                                     intel::scheduler_target_fmax_mhz(480)]] {
        unsigned hash = 0;
        for (size_t i = 0; i < size; i++) {
          hash = (hash * kSeed) + input_a[i];
        }
        output_a[1] = hash;
      });
    });

    q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_a(output_buffer, h, write_only, no_init);

      h.single_task<Fmax240>([=]() [[intel::kernel_args_restrict,
                                     intel::scheduler_target_fmax_mhz(240)]] {
        unsigned hash = 0;
        for (size_t i = 0; i < size; i++) {
          hash = (hash * kSeed) + input_a[i];
        }
        output_a[2] = hash;
      });
    });

    q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_a(output_buffer, h, write_only, no_init);

      h.single_task<Fmax240II>([=]() [[intel::kernel_args_restrict,
                                       intel::scheduler_target_fmax_mhz(240)]] {
        unsigned hash = 0;
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (size_t i = 0; i < size; i++) {
          hash = (hash * kSeed) + input_a[i];
        }
        output_a[3] = hash;
      });
    });

  } catch (exception const &e) {
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

inline unsigned BKDRHashGolden(std::vector<char> input_data) {
  unsigned hash = 0;
  for (int i = 0; i < input_data.size(); ++i) {
    hash = (hash * kSeed) + input_data[i];
  }
  return hash;
}

int main() {
  // input string "qr6KUBBmLtVUlX9"
  std::vector<char> input_data = {'q', 'r', '6', 'K', 'U', 'B', 'B', 'm',
                                  'L', 't', 'V', 'U', 'l', 'X', '9'};
  std::vector<unsigned> output_data(4);

  KernelRun(input_data.size(), input_data, output_data);

  bool passed = true;
  unsigned golden = BKDRHashGolden(input_data);
  if (output_data[0] != golden) {
    std::cout << "Kernel Default Output Mismatch: \n"
              << "output = " << output_data[0] << ", golden = " << golden
              << "\n";
    passed = false;
  }
  if (output_data[1] != golden) {
    std::cout << "Kernel Fmax480 Output Mismatch: \n"
              << "output = " << output_data[1] << ", golden = " << golden
              << "\n";
    passed = false;
  }
  if (output_data[2] != golden) {
    std::cout << "Kernel Fmax240 Output Mismatch: \n"
              << "output = " << output_data[2] << ", golden = " << golden
              << "\n";
    passed = false;
  }
  if (output_data[3] != golden) {
    std::cout << "Kernel Fmax240II Output Mismatch: \n"
              << "output = " << output_data[3] << ", golden = " << golden
              << "\n";
    passed = false;
  }

  if (passed) {
    std::cout << "PASSED: all kernel results are correct.\n";
  } else {
    std::cout << "FAILED\n";
  }
  return passed ? 0 : 1;
}
