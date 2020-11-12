//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <vector>

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

// problem input size
constexpr size_t kInSize = 1000000;
constexpr double kInputMB = (kInSize * sizeof(int)) / (1024 * 1024);
constexpr int kRandMax = 7777;

// Forward declare the kernel names
// (This will become unnecessary in a future compiler version.)
class KernelArgsRestrict;
class KernelArgsNoRestrict;

// Return the execution time of the event, in seconds
double GetExecutionTime(const event &e) {
  double start_k = e.get_profiling_info<info::event_profiling::command_start>();
  double end_k = e.get_profiling_info<info::event_profiling::command_end>();
  double kernel_time = (end_k - start_k) * 1e-9; // ns to s
  return kernel_time;
}

void RunKernels(size_t size, std::vector<int> &in, std::vector<int> &nr_out,
                std::vector<int> &r_out) {

#if defined(FPGA_EMULATOR)
  INTEL::fpga_emulator_selector device_selector;
#else
  INTEL::fpga_selector device_selector;
#endif

  try {
    // create the SYCL device queue
    queue q(device_selector, dpc_common::exception_handler,
            property::queue::enable_profiling{});

    buffer in_buf(in);
    // Use verbose SYCL 1.2 syntax for the output buffer.
    // (This will become unnecessary in a future compiler version.)
    buffer<int, 1> nr_out_buf(nr_out.data(), size);
    buffer<int, 1> r_out_buf(r_out.data(), size);

    // submit the task that DOES NOT apply the kernel_args_restrict attribute
    auto e_nr = q.submit([&](handler &h) {
      auto in_acc = in_buf.get_access<access::mode::read>(h);
      auto out_acc = nr_out_buf.get_access<access::mode::discard_write>(h);

      h.single_task<KernelArgsNoRestrict>([=]() {
        for (size_t i = 0; i < size; i++) {
          out_acc[i] = in_acc[i];
        }
      });
    });

    // submit the task that DOES apply the kernel_args_restrict attribute
    auto e_r = q.submit([&](handler &h) {
      auto in_acc = in_buf.get_access<access::mode::read>(h);
      auto out_acc = r_out_buf.get_access<access::mode::discard_write>(h);

      h.single_task<KernelArgsRestrict>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < size; i++) {
          out_acc[i] = in_acc[i];
        }
      });
    });

    // measure the execution time of each kernel
    double nr_time = GetExecutionTime(e_nr);
    double r_time = GetExecutionTime(e_r);

    std::cout << "Kernel throughput without attribute: " << (kInputMB / nr_time)
              << " MB/s\n";
    std::cout << "Kernel throughput with attribute: " << (kInputMB / r_time)
              << " MB/s\n";

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cout << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.get_cl_code() == CL_DEVICE_NOT_FOUND) {
      std::cout << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cout << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }
}

int main() {
  // seed the random number generator
  srand(0);

  // input/output data
  std::vector<int> in(kInSize);
  std::vector<int> nr_out(kInSize), r_out(kInSize);

  // generate some random input data
  for (size_t i = 0; i < kInSize; i++) {
    in[i] = rand() % kRandMax;
  }

  // Run the kernels
  RunKernels(kInSize, in, nr_out, r_out);

  // validate the results
  for (size_t i = 0; i < kInSize; i++) {
    if (in[i] != nr_out[i]) {
      std::cout << "FAILED: mismatch at entry " << i
                << " of 'KernelArgsNoRestrict' kernel output\n";
      return 1;
    }
  }
  for (size_t i = 0; i < kInSize; i++) {
    if (in[i] != r_out[i]) {
      std::cout << "FAILED: mismatch at entry " << i
                << " of 'KernelArgsRestrict' kernel output\n";
      return 1;
    }
  }

  std::cout << "PASSED\n";

  return 0;
}
