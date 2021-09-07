//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class NoSchedulerTargetFMAX;
class SchedulerTargetFMAX;

// BKDR hash function
unsigned int BKDRHash(const char *str, unsigned int length) {
  unsigned int seed = 1313;
  unsigned int hash = 0;
  unsigned int i = 0;
  for (i = 0; i < length; ++str, ++i) {
    hash = (hash * seed) + (*str);
  }
  return hash;
}

// Runs the Kernel
void KernelRun(size_t size, const std::vector<std::string> &input_data,
               std::vector<unsigned> &output_data_n,
               std::vector<unsigned> &output_data) {

#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif

  try {
    // create the SYCL device queue
    queue q(device_selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer input_buffer(input_data);
    buffer output_n_buffer(output_data_n);
    buffer output_buffer(output_data);

    auto e_g = q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_a(output_n_buffer, h, write_only, no_init);

      h.single_task<NoSchedulerTargetFMAX>([=
      ]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < size; i++) {
          output_a[i] = BKDRHash(input_a[i].c_str(), input_a[i].length());
        }
      });
    });

    auto e_l = q.submit([&](handler &h) {
      accessor input_a(input_buffer, h, read_only);
      accessor output_a(output_buffer, h, write_only, no_init);

      h.single_task<SchedulerTargetFMAX>([=
      ]() [[intel::kernel_args_restrict,
            intel::scheduler_target_fmax_mhz(480)]] {
        for (size_t i = 0; i < size; i++) {
          output_a[i] = BKDRHash(input_a[i].c_str(), input_a[i].length());
        }
      });
    });

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

int main() {
  std::vector<std::string> input_data = {"VwEEB",           "P7pX1",
                                         "jgwkwWV4vc",      "Hja9fr3u4x",
                                         "qr6KUBBmLtVUlX9", "si9NUUs6ghvcxBj"};
  std::vector<unsigned> output_data_n(input_data.size()),
      output_data(input_data.size());

  KernelRun(input_data.size(), input_data, output_data_n, output_data);

  bool passed = true;
  for (size_t i = 0; i < input_data.size(); i++) {
    unsigned golden = BKDRHash(input_data[i].c_str(), input_data[i].length());
    if (output_data_n[i] != golden) {
      std::cout << "Output Mismatch: \n"
                << "output_data_n[" << i << "] = " << output_data_n[i] << "\n"
                << "golden = " << golden << "\n";
      passed = false;
    }
    if (output_data[i] != golden) {
      std::cout << "Output Mismatch: \n"
                << "output_data[" << i << "] = " << output_data[i] << "\n"
                << "golden = " << golden << "\n";
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

