//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// Matrix dimensions
constexpr size_t kNumRows = 4;
constexpr size_t kNumCols = 4;
constexpr size_t kNumElements = kNumRows * kNumCols;

// Total floating point ops performed by the kernel
constexpr size_t kTotalOps = (4 + (3*kNumCols)) * kNumElements;


// Forward declare the kernel name
// (This will become unnecessary in a future compiler version.)
template <int N> class KernelCompute;

// The kernel implements a matrix multiplication.
// This is not meant to be a high performance implementation on FPGA!
// It's just a simple kernel with nested loops to illustrate loop coalescing.
template <int coalesce_factor>
void MatrixMultiply(const device_selector &selector,
                    const std::vector<float> &matrix_a,
                    const std::vector<float> &matrix_b,
                    std::vector<float> &res) {
  double kernel_time = 0.0;
  try {
    auto prop_list = property_list{property::queue::enable_profiling()};

    queue q(selector, dpc_common::exception_handler, prop_list);

    buffer buffer_in_a(matrix_a);
    buffer buffer_in_b(matrix_b);
    // Use verbose SYCL 1.2 syntax for the output buffer.
    // (This will become unnecessary in a future compiler version.)
    buffer<float, 1> buffer_out(res.data(), kNumElements);

    event e = q.submit([&](handler &h) {
      auto accessor_matrix_a = buffer_in_a.get_access<access::mode::read>(h);
      auto accessor_matrix_b = buffer_in_b.get_access<access::mode::read>(h);
      auto accessor_res = buffer_out.get_access<access::mode::discard_write>(h);

      // The kernel_args_restrict promises the compiler that this kernel's
      // accessor arguments won't alias (i.e. non-overlapping memory regions).
      h.single_task<class KernelCompute<coalesce_factor>>(
                                       [=]() [[intel::kernel_args_restrict]] {
        size_t idx = 0;
        float a[kNumRows][kNumCols];
        float b[kNumRows][kNumCols];
        float tmp[kNumRows][kNumCols];

        // The loop_coalesce instructs the compiler to attempt to "merge"
        // coalesce_factor loop levels of this nested loop together.
        // For example, a coalesce_factor of 2 turns this into a single loop.
        [[intelfpga::loop_coalesce(coalesce_factor)]]
        for (size_t i = 0; i < kNumRows; ++i) {
          for (size_t j = 0; j < kNumCols; ++j) {
            a[i][j] = accessor_matrix_a[idx];
            b[i][j] = accessor_matrix_b[idx];
            tmp[i][j] = 0.0;
            idx++;
          }
        }

        // Applying loop_coalesce to the outermost loop of a deeply nested
        // loop results coalescing from the outside in.
        // For example, a coalesce_factor of 2 coalesces the "i" and "j" loops,
        // making a doubly nested loop.
        [[intelfpga::loop_coalesce(coalesce_factor)]]
        for (size_t i = 0; i < kNumRows; ++i) {
          for (size_t j = 0; j < kNumCols; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < kNumCols; ++k) {
              sum += a[i][k] * b[k][j];
            }
            tmp[i][j] = sum;
          }
        }

        idx = 0;
        [[intelfpga::loop_coalesce(coalesce_factor)]]
        for (size_t i = 0; i < kNumRows; ++i) {
          for (size_t j = 0; j < kNumCols; ++j) {
            accessor_res[idx] = tmp[i][j];
            idx++;
          }
        }

      });
    });

    // Kernel profiling data
    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();
    // convert nanoseconds to microseconds
    kernel_time = (double)(end - start) * 1e-3;

  } catch (exception const &exc) {
    std::cout << "Caught synchronous SYCL exception:\n" << exc.what() << '\n';
    if (exc.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  std::cout << "Loop Coalesce: " << coalesce_factor
            << " -- kernel time : " << kernel_time << " microseconds\n";
  std::cout << "Throughput for kernel with coalesce_factor " << coalesce_factor
            << ": ";
  std::cout << std::fixed << std::setprecision(0)
            << (((double)kTotalOps * sizeof(float) * 1e-3f) /
                (kernel_time * 1e-6f)) << "KB/s\n";
}

int main() {
  std::vector<float> matrix_a(kNumElements);
  std::vector<float> matrix_b(kNumElements);
  std::vector<float> matrix_output_no_col(kNumElements);
  std::vector<float> matrix_output(kNumElements);

  // Specify the matrices to be multiplied
  for (size_t i = 0; i < kNumRows; i++) {
    size_t pos = i * kNumCols;
    // Initialize A as identity matrix
    matrix_a[i + pos] = 1.0;
    for (size_t j = 0; j < kNumCols; j++) {
      matrix_b[pos + j] = i * j + 1;
    }
  }

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector selector;
#else
  INTEL::fpga_selector selector;
#endif

  // Two versions of the simple matrix multiply kernel will be enqueued:
  //  - with coalesce_factor=1 (i.e. no loop coalescing)
  //  - with coalesce_factor=2 (coalesce two nested levels)
  MatrixMultiply<1>(selector, matrix_a, matrix_b, matrix_output_no_col);
  MatrixMultiply<2>(selector, matrix_a, matrix_b, matrix_output);

  // Correctness check
  bool passed = true;
  for (size_t i = 0; i < kNumRows; i++) {
    size_t pos = i * kNumCols;
    for (size_t j = 0; j < kNumCols; j++) {
      float val_noCol = matrix_output_no_col[pos + j];
      float val = matrix_output[pos + j];
      if (val_noCol != i * j + 1 || val != i * j + 1) {
        std::cout << "FAILED: The results are incorrect\n";
        passed = false;
      }
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
