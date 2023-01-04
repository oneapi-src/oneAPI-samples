//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iomanip>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

#if defined(FPGA_SIMULATOR)
constexpr size_t kTripCount{100};
#else
constexpr size_t kTripCount{10000000};
#endif
constexpr size_t kDifferentTripCount{kTripCount + 1};
constexpr size_t kArraySize{100};

using FixedArray = std::array<int, kArraySize>;

using namespace sycl;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class DefaultFusionKernel;
class NoFusionKernel;
class DefaultNoFusionKernel;
class FusionFunctionKernel;

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
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

// Returns kernel runtime in ns
auto KernelRuntime(event e) {
  auto start{e.get_profiling_info<info::event_profiling::command_start>()};
  auto end{e.get_profiling_info<info::event_profiling::command_end>()};

  return (end - start);
}

// Fuses inner loops by default, since the trip counts are equal, and there are
// no dependencies between loops
void DefaultFusion(FixedArray &m_array_1, FixedArray &m_array_2) {
  try {
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    buffer buff_1(m_array_1);
    buffer buff_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor a_1(buff_1, h, write_only, no_init);
      accessor a_2(buff_2, h, write_only, no_init);

      h.single_task<DefaultFusionKernel>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        for (size_t i = 0; i < kTripCount; i++) {
          a_1[i % kArraySize] = i % kArraySize;
        }
        for (size_t i = 0; i < kTripCount; i++) {
          a_2[i % kArraySize] = i % kArraySize;
        }
      });
    });

    auto kernel_time = KernelRuntime(e);

    // Kernel consists of two loops with one array assignment and two modulos in
    // each.
    int num_ops_per_kernel = 6 * kTripCount;
    std::cout << "Throughput for kernel with default loop fusion and with "
                 "equally-sized loops: "
              << ((double)num_ops_per_kernel / kernel_time) << " Ops/ns\n";

  } catch (sycl::exception const &e) {
    ErrorReport(e);
  }
}

// Does not fuse inner loops because of the [[intel::nofusion]] attribute. Were
// this attribute not present, the loops would fuse by default.
void NoFusion(FixedArray &m_array_1, FixedArray &m_array_2) {
  try {
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});

    buffer buff_1(m_array_1);
    buffer buff_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor a_1(buff_1, h, write_only, no_init);
      accessor a_2(buff_2, h, write_only, no_init);

      h.single_task<NoFusionKernel>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        [[intel::nofusion]]                  // NO-FORMAT: Attribute
        for (size_t i = 0; i < kTripCount; i++) {
          a_1[i % kArraySize] = i % kArraySize;
        }
        for (size_t i = 0; i < kTripCount; i++) {
          a_2[i % kArraySize] = i % kArraySize;
        }

      });
    });

    auto kernel_time = KernelRuntime(e);

    // Kernel consists of two loops with one array assignment and two modulos in
    // each.
    int num_ops_per_kernel = 6 * kTripCount;
    std::cout << "Throughput for kernel with the nofusion attribute and with "
                 "equally-sized loops: "
              << ((double)num_ops_per_kernel / kernel_time) << " Ops/ns\n";

  } catch (sycl::exception const &e) {
    ErrorReport(e);
  }
}

// Does not fuse inner loops by default, since the trip counts are different.
void DefaultNoFusion(FixedArray &m_array_1, FixedArray &m_array_2) {
  try {
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});

    buffer buff_1(m_array_1);
    buffer buff_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor a_1(buff_1, h, write_only, no_init);
      accessor a_2(buff_2, h, write_only, no_init);

      h.single_task<DefaultNoFusionKernel>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        // Different tripcounts, does not fuse by default
        for (size_t i = 0; i < kDifferentTripCount + 1; i++) {
          a_1[i % kArraySize] = i % kArraySize;
        }
        for (size_t i = 0; i < kTripCount; i++) {
          a_2[i % kArraySize] = i % kArraySize;
        }

      });
    });

    auto kernel_time = KernelRuntime(e);

    // Kernel consists of two loops different trip counts, each with one array
    // assignment and two modulos.
    int num_ops_per_kernel = 3 * kDifferentTripCount + 3 * kTripCount;
    std::cout << "Throughput for kernel without fusion by default with "
                 "unequally-sized loops: "
              << ((double)num_ops_per_kernel / kernel_time) << " Ops/ns\n";

  } catch (sycl::exception const &e) {
    ErrorReport(e);
  }
}

// The compiler is explicitly told to fuse the inner loops using the
// fpga_loop_fuse<>() function. Were this function not used, the loops would not
// fuse by default, since the trip counts of the loops are different.
void FusionFunction(FixedArray &m_array_1, FixedArray &m_array_2) {
  try {
    queue q(selector, fpga_tools::exception_handler,
            property::queue::enable_profiling{});

    buffer buff_1(m_array_1);
    buffer buff_2(m_array_2);

    event e = q.submit([&](handler &h) {
      accessor a_1(buff_1, h, write_only, no_init);
      accessor a_2(buff_2, h, write_only, no_init);

      h.single_task<FusionFunctionKernel>([=
      ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
        sycl::ext::intel::fpga_loop_fuse([&] {
          // Different tripcounts, does not fuse by default
          for (size_t i = 0; i < kDifferentTripCount; i++) {
            a_1[i % kArraySize] = i % kArraySize;
          }
          for (size_t i = 0; i < kTripCount; i++) {
            a_2[i % kArraySize] = i % kArraySize;
          }
        });
      });
    });

    auto kernel_time = KernelRuntime(e);
    // Kernel consists of two loops different trip counts, each with one array
    // assignment and two modulos.
    int num_ops_per_kernel = 3 * kDifferentTripCount + 3 * kTripCount;
    std::cout << "Throughput for kernel with a loop fusion function with "
                 "unequally-sized loops: "
              << ((double)num_ops_per_kernel / kernel_time) << " Ops/ns\n";

  } catch (sycl::exception const &e) {
    ErrorReport(e);
  }
}

int main() {
  // Arrays will be populated in kernel loops
  FixedArray default_fusion_1;
  FixedArray default_fusion_2;

  FixedArray no_fusion_1;
  FixedArray no_fusion_2;

  FixedArray default_nofusion_1;
  FixedArray default_nofusion_2;

  FixedArray fusion_function_1;
  FixedArray fusion_function_2;

  // Instantiate kernel logic with and without loop fusion to compare
  // performance
  DefaultFusion(default_fusion_1, default_fusion_2);
  NoFusion(no_fusion_1, no_fusion_2);
  DefaultNoFusion(default_nofusion_1, default_nofusion_2);
  FusionFunction(fusion_function_1, fusion_function_2);

  // Verify results: arrays should be equal, and the i^th element shoul equal i
  for (size_t i = 0; i < kArraySize; i++) {
    if (default_fusion_1[i] != default_fusion_2[i] ||
        default_fusion_1[i] != i) {
      std::cout << "FAILED: The DefaultFusionKernel results are incorrect"
                << '\n';
      return 1;
    }
  }
  for (size_t i = 0; i < kArraySize; i++) {
    if (no_fusion_1[i] != no_fusion_2[i] || no_fusion_1[i] != i) {
      std::cout << "FAILED: The NoFusionKernel results are incorrect" << '\n';
      return 1;
    }
  }

  for (size_t i = 0; i < kArraySize; i++) {
    if (default_nofusion_1[i] != default_nofusion_2[i] ||
        default_nofusion_1[i] != i) {
      std::cout << "FAILED: The DefaultNoFusionKernel results are incorrect"
                << '\n';
      return 1;
    }
  }

  for (size_t i = 0; i < kArraySize; i++) {
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
