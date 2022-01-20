//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <chrono>
#include <numeric>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

constexpr size_t vector_size = 1000000; // size of input vectors
constexpr double kNs = 1e9;             // number of nanoseconds in a second

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class VecAdd;

event runVecAdd(sycl::queue &q, const std::vector<int> &a_vec,
                const std::vector<int> &b_vec, const std::vector<int> &c_vec,
                std::vector<int> &sum_vec) {
#if defined(NO_INTERLEAVING) && defined(TWO_CHANNELS)
  buffer a_buf(a_vec, {property::buffer::mem_channel{1}});
  buffer b_buf(b_vec, {property::buffer::mem_channel{2}});
  buffer c_buf(c_vec, {property::buffer::mem_channel{1}});
  buffer sum_buf(sum_vec, {property::buffer::mem_channel{2}});
#elif defined(NO_INTERLEAVING) && defined(FOUR_CHANNELS)
  buffer a_buf(a_vec, {property::buffer::mem_channel{1}});
  buffer b_buf(b_vec, {property::buffer::mem_channel{2}});
  buffer c_buf(c_vec, {property::buffer::mem_channel{3}});
  buffer sum_buf(sum_vec, {property::buffer::mem_channel{4}});
#else
  buffer a_buf(a_vec);
  buffer b_buf(b_vec);
  buffer c_buf(c_vec);
  buffer sum_buf(sum_vec);
#endif

  event e = q.submit([&](handler &h) {
    accessor a(a_buf, h, read_only);
    accessor b(b_buf, h, read_only);
    accessor c(c_buf, h, read_only);
    accessor sum(sum_buf, h, write_only, no_init);

    h.single_task<VecAdd>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < vector_size; i++)
        sum[i] = a[i] + b[i] + c[i];
    });
  });
  return e;
}

int main() {
  // Host and kernel profiling
  event e;
  ulong t1_kernel, t2_kernel;
  double time_kernel;

  // Create input and output vectors
  std::vector<int> a, b, c, sum_fpga, sum_cpu;
  a.resize(vector_size);
  b.resize(vector_size);
  c.resize(vector_size);
  sum_fpga.resize(vector_size);
  sum_cpu.resize(vector_size);

  // Initialize input vectors with values from 0 to vector_size - 1
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 0);
  std::iota(c.begin(), c.end(), 0);

// Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif
  try {
    auto prop_list =
        sycl::property_list{sycl::property::queue::enable_profiling()};

    sycl::queue q(device_selector, dpc_common::exception_handler, prop_list);

    std::cout << "\nVector size: " << vector_size << "\n";

    e = runVecAdd(q, a, b, c, sum_fpga);
    e.wait();

    // Compute kernel execution time
    t1_kernel = e.get_profiling_info<info::event_profiling::command_start>();
    t2_kernel = e.get_profiling_info<info::event_profiling::command_end>();
    time_kernel = (t2_kernel - t1_kernel) / kNs;

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

  // Compute the reference solution
  for (int i = 0; i < vector_size; ++i)
    sum_cpu[i] = a[i] + b[i] + c[i];

  // Verify output and print pass/fail
  bool passed = true;
  int num_errors = 0;
  for (int b = 0; b < vector_size; b++) {
    if (num_errors < 10 && sum_fpga[b] != sum_cpu[b]) {
      passed = false;
      std::cerr << " (mismatch, expected " << sum_cpu[b] << ")\n";
      num_errors++;
    }
  }

  if (passed) {
    std::cout << "Verification PASSED\n\n";

    // Report host execution time and throughput
    std::cout.setf(std::ios::fixed);

    // Input size in MB
    constexpr double num_mb = (vector_size * sizeof(uint32_t)) / (1024 * 1024);

    // Report kernel execution time and throughput
    std::cout << "Kernel execution time: " << time_kernel << " seconds\n";
#if !defined(NO_INTERLEAVING)
    std::cout << "Kernel throughput: " << (num_mb / time_kernel) << " MB/s\n\n";
#else
    std::cout << "Kernel throughput without burst-interleaving: "
              << (num_mb / time_kernel) << " MB/s\n\n";
#endif
  } else {
    std::cerr << "Verification FAILED\n";
    return 1;
  }
  return 0;
}
