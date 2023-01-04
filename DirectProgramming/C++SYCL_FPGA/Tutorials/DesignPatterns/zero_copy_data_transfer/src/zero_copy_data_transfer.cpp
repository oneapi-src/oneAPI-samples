#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"

#include "buffer_kernel.hpp"
#include "zero_copy_kernel.hpp"

using namespace sycl;

// the data type - assert that it is arithmetic
// this allows the design to easily switch between types by changing
// the line below
using Type = int;
static_assert(std::is_arithmetic<Type>::value);

int main(int argc, char* argv[]) {
  // parse command line arguments
#if defined(FPGA_EMULATOR)
  size_t size = 10000;
  size_t iterations = 1;
#elif FPGA_SIMULATOR
  size_t size = 700;
  size_t iterations = 1;
#else
  size_t size = 100000000;
  size_t iterations = 5;
#endif

  // Allow the size to be changed by a command line argument
  if (argc > 1) {
    size = atoi(argv[1]);
  }

  // check the size
  if (size <= 0) {
    std::cerr << "ERROR: size must be greater than 0\n";
    return 1;
  }

  try {
    // device selector
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // create the device queue
    queue q(selector, fpga_tools::exception_handler);

    // make sure the device supports USM host allocations
    auto device = q.get_device();
    if (!device.get_info<info::device::usm_host_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
      return 1;
    }

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // the golden output
    std::vector<Type> out_gold(size);

    // input and output data for the buffer version
    std::vector<Type> in_buffer(size), out_buffer(size);

    // input and output data for the zero-copy version
    // malloc_host allocates memory specifically in the host's address space
    Type* in_zero_copy = malloc_host<Type>(size, q.get_context());
    Type* out_zero_copy = malloc_host<Type>(size, q.get_context());
    
    // ensure that we could allocate space for both the input and output
    if (in_zero_copy == NULL) {
      std::cerr << "ERROR: failed to allocate space for 'in_zero_copy'\n";
      return 1;
    }
    if (out_zero_copy == NULL) {
      std::cerr << "ERROR: failed to allocate space for 'out_zero_copy'\n";
      return 1;
    }

    // generate some random input data and compute the golden result
    for (int i = 0; i < size; i++) {
      // generate a random value
      Type n = Type(rand() % 100);

      // populate the inputs
      in_buffer[i] = in_zero_copy[i] = n;

      // compute the golden result
      out_gold[i] = n * i;
    }

    // run the buffer version kernel
    std::cout << "Running the buffer kernel version with size=" << size << "\n";
    std::vector<double> buffer_kernel_latency(iterations);
    for (size_t i = 0; i < iterations; i++) {
      buffer_kernel_latency[i] = SubmitBufferKernel<Type>(q, in_buffer,
                                                          out_buffer, size);
    }

    // run the the zero-copy version kernel
    std::cout << "Running the zero-copy kernel version with size=" << size
              << "\n";
    std::vector<double> zero_copy_latency(iterations);
    for (size_t i = 0; i < iterations; i++) {
      zero_copy_latency[i] = SubmitZeroCopyKernel<Type>(q, in_zero_copy,
                                                        out_zero_copy, size);
    }

    // validate the outputs
    // validate the buffer version
    for (int i = 0; i < size; i++) {
      if (out_gold[i] != out_buffer[i]) {
        std::cerr << "ERROR: mismatch at entry " << i
                  << " of 'Buffer' kernel output "
                  << "(" << out_gold[i] << "," << out_buffer[i] << ")"
                  << "\n";
        return 1;
      }
    }
    // validate the the zero-copy version
    for (int i = 0; i < size; i++) {
      if (out_gold[i] != out_zero_copy[i]) {
        std::cerr << "ERROR: mismatch at entry " << i
                  << " of 'ZeroCopy' kernel output "
                  << "(" << out_gold[i] << "," << out_zero_copy[i] << ")"
                  << "\n";
        return 1;
      }
    }

    // The FPGA emulator or simulator do not accurately represent the hardware performance
    // so we don't print performance results when running with the emulator or simulator
#ifdef FPGA_EMULATOR
#elif FPGA_SIMULATOR
#else
    // Compute the average latency across all iterations.
    // We use the first iteration as a 'warmup' for the FPGA,
    // so we ignore its results.
    double buffer_avg_lat = std::accumulate(buffer_kernel_latency.begin() + 1,
                                            buffer_kernel_latency.end(), 0.0) /
                                        (iterations - 1);
    double zero_copy_avg_lat =
        std::accumulate(zero_copy_latency.begin() + 1,
                        zero_copy_latency.end(), 0.0) /
                    (iterations - 1);

    std::cout << "Average latency for the buffer kernel: " << buffer_avg_lat
              << " ms\n";
    std::cout << "Average latency for the zero-copy kernel: "
              << zero_copy_avg_lat << " ms\n";
#endif

    // free the USM host allocations
    // note that these are calls to sycl::free()
    free(in_zero_copy, q);
    free(out_zero_copy, q);

  } catch (exception const& e) {
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

  std::cout << "PASSED\n";

  return 0;
}
