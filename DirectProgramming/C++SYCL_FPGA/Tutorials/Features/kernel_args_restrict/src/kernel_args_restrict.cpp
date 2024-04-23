//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <algorithm>
#include <numeric>
#include <vector>

#include "exception_handler.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
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

    buffer in_buf(in);
    buffer nr_out_buf(nr_out);
    buffer r_out_buf(r_out);

    // submit the task that DOES NOT apply the kernel_args_restrict attribute
    auto e_nr = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(nr_out_buf, h, write_only, no_init);

      h.single_task<KernelArgsNoRestrict>([=]() {
        for (size_t i = 0; i < size; i++) {
          out_acc[i] = in_acc[i];
        }
      });
    });

    // submit the task that DOES apply the kernel_args_restrict attribute
    auto e_r = q.submit([&](handler &h) {
      accessor in_acc(in_buf, h, read_only);
      accessor out_acc(r_out_buf, h, write_only, no_init);

      h.single_task<KernelArgsRestrict>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < size; i++) {
          out_acc[i] = in_acc[i];
        }
      });
    });

    // measure the execution time of each kernel
    double size_mb = (size * sizeof(int)) / (1024 * 1024);
    double nr_time = GetExecutionTime(e_nr);
    double r_time = GetExecutionTime(e_r);

    std::cout << "Size of vector: " << size << " elements\n";
    std::cout << "Kernel throughput without attribute: " << (size_mb / nr_time)
              << " MB/s\n";
    std::cout << "Kernel throughput with attribute: " << (size_mb / r_time)
              << " MB/s\n";

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

  // Exiting the 'try' scope above, where the buffers were declared, will cause 
  // the buffer destructors to be called which will wait until the kernels that
  // use them to finish and copy the data back to the host (if the buffer was
  // written to).
  // Therefore, at this point in the code, we know that the kernels have
  // finished and the data has been transferred back to the host (in the
  // 'nr_out' and 'r_out' vectors).
}

int main(int argc, char* argv[]) {
  // size of vectors to copy, allow user to change it from the command line
#if defined(FPGA_SIMULATOR)
  size_t size = 5000; // smaller size to keep the default runtime reasonable
#else
  size_t size = 5000000;
#endif

  if (argc > 1) size = atoi(argv[1]);

  // input/output data
  std::vector<int> in(size);
  std::vector<int> nr_out(size), r_out(size);

  // generate some input data
  std::iota(in.begin(), in.end(), 0);

  // clear the output data
  std::fill(nr_out.begin(), nr_out.end(), -1);
  std::fill(r_out.begin(), r_out.end(), -1);

  // Run the kernels
  RunKernels(size, in, nr_out, r_out);

  // validate the results
  bool passed = true;

  for (size_t i = 0; i < size; i++) {
    if (in[i] != nr_out[i]) {
      std::cout << "FAILED: mismatch at entry " << i
                << " of 'KernelArgsNoRestrict' kernel output\n";
      std::cout << " (" << in[i] << " != " << nr_out[i]
                << ", in[i] != out[i])\n";
      passed = false;
      break;
    }
  }
  
  for (size_t i = 0; i < size; i++) {
    if (in[i] != r_out[i]) {
      std::cout << "FAILED: mismatch at entry " << i
                << " of 'KernelArgsRestrict' kernel output\n";
      std::cout << " (" << in[i] << " != " << r_out[i]
                << ", in[i] != out[i])\n";

      passed = false;
      break;
    }
  }

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 0;
  }
}
