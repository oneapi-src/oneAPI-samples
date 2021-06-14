//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <iomanip>
#include <iostream>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class KernelComputeStallFree;
class KernelComputeStallEnable;

constexpr int kSeed = 0;
constexpr int kWork = 500;
constexpr int kNumElements = 1000;
constexpr int kTotalOps = kNumElements * kWork;

typedef float WorkType;
typedef std::vector<WorkType> WorkVec;

static WorkType RealWork(WorkType a, WorkType b) {
  // This will create a large cluster, which will exaggerate
  // the effect of the stall-enable cluster.
#pragma unroll
  for (size_t j = 0; j < kWork; j++) {
    if (j & 1)
      a -= b;
    else
      a *= b;
  }
  return a;
}

typedef accessor<WorkType, 1, access::mode::read, access::target::global_buffer> ReadAccessor;
typedef accessor<WorkType, 1, access::mode::write, access::target::global_buffer> WriteAccessor;

static void Work(const ReadAccessor &vec_a,
                 const ReadAccessor &vec_b,
                 const WriteAccessor & vec_res) {
  for (size_t idx = 0; idx < kNumElements; idx++) {
    auto a = vec_a[idx];
    auto b = vec_b[idx];
    vec_res[idx] = RealWork(a, b);
  }
}

void DoSomeWork(const device_selector &selector, const WorkVec &vec_a,
                const WorkVec &vec_b, WorkVec &res) {
  double kernel_time = 0.0;
  try {
    auto prop_list = property_list{property::queue::enable_profiling()};

    queue q(selector, dpc_common::exception_handler, prop_list);

    buffer buffer_in_a(vec_a);
    buffer buffer_in_b(vec_b);
    buffer buffer_out(res);

    event e = q.submit([&](handler &h) {
      accessor accessor_vec_a(buffer_in_a, h, read_only);
      accessor accessor_vec_b(buffer_in_b, h, read_only);
      accessor accessor_res(buffer_out, h, write_only, noinit);

      // The kernel_args_restrict promises the compiler that this kernel's
      // accessor arguments won't alias (i.e. non-overlapping memory regions).
#ifdef STALL_FREE
        h.single_task<class KernelComputeStallFree>(
                                         [=]() [[intel::kernel_args_restrict]] {
          // Run using a function with stall free clusters
          Work(accessor_vec_a, accessor_vec_b, accessor_res);
        });
#else // STALL_FREE
        h.single_task<class KernelComputeStallEnable>(
                                         [=]() [[intel::kernel_args_restrict, intel::use_stall_enable_clusters]] {
          // Run using a function with stall enable clusters
          Work(accessor_vec_a, accessor_vec_b, accessor_res);
        });
#endif // STALL_FREE
      });

    // Kernel profiling data
    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();
    // convert nanoseconds to microseconds
    kernel_time = (double)(end - start) * 1e-3;

  } catch (exception const &exc) {
    std::cerr << "Caught synchronous SYCL exception:\n" << exc.what() << '\n';
    if (exc.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

#ifdef STALL_FREE
  std::cout << "Stall free"
#else
  std::cout << "Stall enable"
#endif
            << " Kernel -- kernel time : " << kernel_time << " microseconds\n";
  std::cout << "Throughput for kernel: ";
  std::cout << std::fixed << std::setprecision(0)
            << (((double)kTotalOps * sizeof(WorkType) * 1e-3f) /
                (kernel_time * 1e-6f)) << "KB/s\n";
}

int main() {
  WorkVec vec_a(kNumElements);
  WorkVec vec_b(kNumElements);
  WorkVec vec_output(kNumElements);
  WorkVec vec_expected(kNumElements);

  // Populate the vectors
  srand(kSeed);
  for (size_t i = 0; i < kNumElements; i++) {
    vec_a[i] = static_cast<WorkType>(rand());
    vec_b[i] = static_cast<WorkType>(rand());
    vec_expected[i] = RealWork(vec_a[i], vec_b[i]);
  }

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  DoSomeWork(selector, vec_a, vec_b, vec_output);

  // Correctness check
  bool passed = true;
  for (size_t i = 0; i < kNumElements; i++) {
    auto val = vec_output[i];
    if (val != vec_expected[i]) {
      std::cout << "FAILED: The results are incorrect\n"
                << "Expected: " << vec_expected[i]
                << ", result: " << val << '\n';
      passed = false;
    }
  }

  if (passed) {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return -1;
  }
}
