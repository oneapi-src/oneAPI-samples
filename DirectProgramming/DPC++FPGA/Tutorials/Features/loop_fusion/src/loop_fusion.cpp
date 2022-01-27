//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

constexpr size_t kN{1024};
constexpr size_t kM{50};

using namespace sycl;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class DefaultFusionKernel;
class NoFusionKernel;
class DefaultNoFusionKernel;
class FusionFunctionKernel;

#if defined(FPGA_EMULATOR)
ext::intel::fpga_emulator_selector selector;
#else
ext::intel::fpga_selector selector;
#endif

// Handles error reporting
void ErrorReport(sycl::exception const &e) {
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

// Returns kernel runtime in ms
auto KernelRuntime(event e) {
  auto start{e.get_profiling_info<info::event_profiling::command_start>()};
  auto end{e.get_profiling_info<info::event_profiling::command_end>()};

  // unit is nano second, convert to ms
  return (end - start) * 1e-6;
}

// Fuses inner loops by default, since the trip counts are equal, and there are
// no dependencies between loops
void DefaultFusion(std::array<int, kM> &m_array_1,
                   std::array<int, kM> &m_array_2) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer buffer_array_1(m_array_1);
    buffer buffer_array_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor accessor_array_1(buffer_array_1, h, write_only, no_init);
      accessor accessor_array_2(buffer_array_2, h, write_only, no_init);

      h.single_task<DefaultFusionKernel>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        for (size_t j = 0; j < kN; j++) {
          for (size_t i = 0; i < kM; i++) {
            accessor_array_1[i] = i;
          }
          for (size_t i = 0; i < kM; i++) {
            accessor_array_2[i] = accessor_array_1[i];
          }
        }
      });
    });

    auto kernel_time = KernelRuntime(e);

    // kernel consists of two loops with one array assignment in each.
    int num_ops_per_kernel = 2 * kN * kM;
    std::cout << "Throughput for kernel with default loop fusion and with "
                 "arrays of size "
              << kM << ": " << ((double)num_ops_per_kernel / kernel_time)
              << " Ops/ms\n";

  } catch (sycl::exception const &e) {
    ErrorReport(e);
  }
}

// Does not fuse inner loops because of the [[intel::nofusion]] attribute. Were this attribute not present, the loops would fuse by default.
void NoFusion(std::array<int, kM> &m_array_1, std::array<int, kM> &m_array_2) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer buffer_array_1(m_array_1);
    buffer buffer_array_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor accessor_array_1(buffer_array_1, h, write_only, no_init);
      accessor accessor_array_2(buffer_array_2, h, write_only, no_init);

      h.single_task<NoFusionKernel>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        for (size_t j = 0; j < kN; j++) {
          [[intel::nofusion]]  // NO-FORMAT: Attribute
          for (size_t i = 0; i < kM; i++) {
            accessor_array_1[i] = i;
          }
          for (size_t i = 0; i < kM; i++) {
            accessor_array_2[i] = accessor_array_1[i];
          }
        }

      });
    });

    auto kernel_time = KernelRuntime(e);

    // kernel consists of two loops with one array assignment in each.
    int num_ops_per_kernel = 2 * kN * kM;
    std::cout << "Throughput for kernel with the nofusion attribute and with "
                 "arrays of size "
              << kM << ": " << ((double)num_ops_per_kernel / kernel_time)
              << " Ops/ms\n";

  } catch (sycl::exception const &e) {
    ErrorReport(e);
  }
}

// Does not fuse inner loops by default, since the trip counts are different.
void DefaultNoFusion(std::array<int, kM + 1> &m_array_1,
                     std::array<int, kM> &m_array_2) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer buffer_array_1(m_array_1);
    buffer buffer_array_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor accessor_array_1(buffer_array_1, h, write_only, no_init);
      accessor accessor_array_2(buffer_array_2, h, write_only, no_init);

      h.single_task<DefaultNoFusionKernel>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        for (size_t j = 0; j < kN; j++) {
          // Different tripcounts, does not fuse by default
          for (size_t i = 0; i < kM + 1; i++) {
            accessor_array_1[i] = i;
          }
          for (size_t i = 0; i < kM; i++) {
            accessor_array_2[i] = accessor_array_1[i];
          }
        }

      });
    });

    auto kernel_time = KernelRuntime(e);

    // kernel consists of two loops with one array assignment in each.
    int num_ops_per_kernel = kN * (2 * kM + 1);
    std::cout << "Throughput for kernel without fusion by default and with "
                 "arrays of sizes "
              << kM + 1 << " and " << kM << ": "
              << ((double)num_ops_per_kernel / kernel_time) << " Ops/ms\n";

  } catch (sycl::exception const &e) {
    ErrorReport(e);
  }
}

// The compiler is explicitly told to fuse the inner loops using the fpga_loop_fuse<>() function. Were this function not used, the loops would not fuse by default, since the trip counts of the loops are different.
void FusionFunction(std::array<int, kM + 1> &m_array_1,
                    std::array<int, kM> &m_array_2) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer buffer_array_1(m_array_1);
    buffer buffer_array_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor accessor_array_1(buffer_array_1, h, write_only, no_init);
      accessor accessor_array_2(buffer_array_2, h, write_only, no_init);

      h.single_task<FusionFunctionKernel>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        sycl::ext::intel::fpga_loop_fuse<2>([&] {
          for (size_t j = 0; j < kN; j++) {
            // Different tripcounts, does not fuse by default
            for (size_t i = 0; i < kM + 1; i++) {
              accessor_array_1[i] = i;
            }
            for (size_t i = 0; i < kM; i++) {
              accessor_array_2[i] = accessor_array_1[i];
            }
          }
        });
      });
    });

    auto kernel_time = KernelRuntime(e);
    // kernel consists of two loops with one array assignment in each.
    int num_ops_per_kernel = kN * (2 * kM + 1);
    std::cout << "Throughput for kernel with the loop fusion function wrapper "
                 "and with arrays of sizes "
              << kM + 1 << " and " << kM << ": "
              << ((double)num_ops_per_kernel / kernel_time) << " Ops/ms\n";

  } catch (sycl::exception const &e) {
    ErrorReport(e);
  }
}


int main() {
  // Arrays will be populated in kernel loops 
  std::array<int, kM> default_fusion_1, default_fusion_2, no_fusion_1,
      no_fusion_2, default_nofusion_2, fusion_function_2;
  std::array<int, kM + 1> default_nofusion_1, fusion_function_1;

  // Instantiate kernel logic with and without loop fusion to compare
  // performance
  DefaultFusion(default_fusion_1, default_fusion_2);
  NoFusion(no_fusion_1, no_fusion_2);
  DefaultNoFusion(default_nofusion_1, default_nofusion_2);
  FusionFunction(fusion_function_1, fusion_function_2);

  // Verify results: first kN elements of arrays should be equal, and equal to i
  for (size_t i = 0; i < kM; i++) {
    if (default_fusion_1[i] != default_fusion_2[i] ||
        default_fusion_1[i] != i) {
      std::cout << "FAILED: The DefaultFusionKernel results are incorrect"
                << '\n';
      return 1;
    }
  }
  for (size_t i = 0; i < kM; i++) {
    if (no_fusion_1[i] != no_fusion_2[i] || no_fusion_1[i] != i) {
      std::cout << "FAILED: The NoFusionKernel results are incorrect" << '\n';
      return 1;
    }
  }

  // Since these have arrays have nonequal sizes, only check that they are equal
  // when i < kM
  for (size_t i = 0; i < kM + 1; i++) {
    if ((default_nofusion_1[i] != default_nofusion_2[i] && i < kM) ||
        default_nofusion_1[i] != i) {
      std::cout << "FAILED: The DefaultNoFusionKernel results are incorrect"
                << '\n';
      return 1;
    }
  }

  for (size_t i = 0; i < kM + 1; i++) {
    if ((fusion_function_1[i] != fusion_function_2[i] && i < kM) ||
        fusion_function_1[i] != i) {
      std::cout << "FAILED: The FusionFunctionKernel results are incorrect"
                << '\n';
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct" << '\n';
  return 0;
}
