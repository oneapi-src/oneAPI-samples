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

constexpr size_t N = 2048;
constexpr size_t M = 5;

using namespace sycl;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class DefaultFusionKernel;
class NoFusionKernel;
class DefaultNoFusionKernel;
class FusionFunctionKernel;

void DefaultFusion(const device_selector &selector,
                   std::array<int, M> &m_array_1, std::array<int, M> &m_array_2,
                   const size_t kInnerIters) {
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
        for (size_t j = 0; j < N; j++) {
          for (size_t i = 0; i < kInnerIters; i++) {
            accessor_array_1[i] = i;
          }
          for (size_t i = 0; i < kInnerIters; i++) {
            accessor_array_2[i] = i;
          }
        }
      });
    });

    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();

    // unit is nano second, convert to ms
    double kernel_time = (end - start) * 1e-6;

    // kernel consists of two loops with one array assignment in each.
    int num_ops_per_kernel = 2 * N * kInnerIters;
    std::cout << "Throughput for kernel with default loop fusion and with "
                 "arrays of size "
              << M << ": " << ((double)num_ops_per_kernel / kernel_time)
              << " Ops/ms\n";

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

void NoFusion(const device_selector &selector, std::array<int, M> &m_array_1,
              std::array<int, M> &m_array_2, const size_t kInnerIters) {
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
        for (size_t j = 0; j < N; j++) {
          [[intel::nofusion]]  // NO-FORMAT: Attribute
          for (size_t i = 0; i < kInnerIters; i++) {
            accessor_array_1[i] = i;
          }
          for (size_t i = 0; i < kInnerIters; i++) {
            accessor_array_2[i] = i;
          }
        }

      });
    });

    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();

    // unit is nano second, convert to ms
    double kernel_time = (end - start) * 1e-6;

    // kernel consists of two loops with one array assignment in each.
    int num_ops_per_kernel = 2 * N * kInnerIters;
    std::cout << "Throughput for kernel with the nofusion attribute and with "
                 "arrays of size "
              << M << ": " << ((double)num_ops_per_kernel / kernel_time)
              << " Ops/ms\n";

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

void DefaultNoFusion(const device_selector &selector,
                     std::array<int, M> &m_array_1,
                     std::array<int, M + 1> &m_array_2, const size_t kInnerIters) {
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
        for (size_t j = 0; j < N; j++) {
          // Different tripcounts, does not fuse by default
          for (size_t i = 0; i < kInnerIters; i++) {
            accessor_array_1[i] = i;
          }
          for (size_t i = 0; i < kInnerIters + 1; i++) {
            accessor_array_2[i] = i;
          }
        }

      });
    });

    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();

    // unit is nano second, convert to ms
    double kernel_time = (end - start) * 1e-6;


    // kernel consists of two loops with one array assignment in each.
    int num_ops_per_kernel = N * (2 * kInnerIters + 1);
    std::cout << "Throughput for kernel without fusion by default and with "
                 "arrays of sizes "
              << M << " and " << M + 1 << ": "
              << ((double)num_ops_per_kernel / kernel_time) << " Ops/ms\n";

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

void FusionFunction(const device_selector &selector,
                    std::array<int, M> &m_array_1,
                    std::array<int, M + 1> &m_array_2, const size_t kInnerIters) {
  try {
    queue q(selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer buffer_array_1(m_array_1);
    buffer buffer_array_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor accessor_array_1(buffer_array_1, h, write_only, no_init);
      accessor accessor_array_2(buffer_array_2, h, write_only, no_init);

      h.single_task<FusionFunctionKernel>([=
      ]() [[intel::kernel_args_restrict,
            intel::loop_fuse(2)]] {  // NO-FORMAT: Attribute
        for (size_t j = 0; j < N; j++) {
          // Different tripcounts, does not fuse by default
          for (size_t i = 0; i < kInnerIters; i++) {
            accessor_array_1[i] = i;
          }
          for (size_t i = 0; i < kInnerIters + 1; i++) {
            accessor_array_2[i] = i;
          }
        }
      });
    });

    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();

    // unit is nano second, convert to ms
    double kernel_time = (end - start) * 1e-6;

    // kernel consists of two loops with one array assignment in each.
    int num_ops_per_kernel = N * (2 * kInnerIters + 1);
    std::cout << "Throughput for kernel with the loop fusion function wrapper "
                 "and with arrays of sizes "
              << M << " and " << M + 1 << ": "
              << ((double)num_ops_per_kernel / kernel_time) << " Ops/ms\n";

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
  std::array<int, M> default_fusion_1, default_fusion_2, no_fusion_1,
      no_fusion_2, fusion_function_1, default_nofusion_1;
  std::array<int, M + 1> fusion_function_2, default_nofusion_2;

#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector selector;
#else
  ext::intel::fpga_selector selector;
#endif

  // Instantiate kernel logic with and without loop fusion to compare
  // performance
  DefaultFusion(selector, default_fusion_1, default_fusion_2, M);
  NoFusion(selector, no_fusion_1, no_fusion_2, M);
  DefaultNoFusion(selector, default_nofusion_1, fusion_function_2, M);
  FusionFunction(selector, fusion_function_1, fusion_function_2, M);

  // Verify results: first N elements of arrays should be equal, and equal to i
  for (size_t i = 0; i < M; i++) {
    if (default_fusion_1[i] != default_fusion_2[i] ||
        default_fusion_1[i] != i) {
      std::cout << "FAILED: The DefaultFusionKernel results are incorrect"
                << '\n';
      return 1;
    }
  }
  for (size_t i = 0; i < M; i++) {
    if (no_fusion_1[i] != no_fusion_2[i] || no_fusion_1[i] != i) {
      std::cout << "FAILED: The NoFusionKernel results are incorrect" << '\n';
      return 1;
    }
  }
  for (size_t i = 0; i < M; i++) {
    if (default_nofusion_1[i] != fusion_function_2[i] ||
        fusion_function_1[i] != i) {
      std::cout << "FAILED: The DefaultNoFusionKernel results are incorrect"
                << '\n';
      return 1;
    }
  }

  for (size_t i = 0; i < M; i++) {
    if (fusion_function_1[i] != fusion_function_2[i] ||
        fusion_function_1[i] != i) {
      std::cout << "FAILED: The FusionFunctionKernel results are incorrect"
                << '\n';
      return 1;
    }
  }
  std::cout << "PASSED: The results are correct" << '\n';
  return 0;
}
