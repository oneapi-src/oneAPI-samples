#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>
#include <type_traits>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#include "buffer_kernel.hpp"
#include "restricted_usm_kernel.hpp"

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
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector selector;
#else
    INTEL::fpga_selector selector;
#endif

    // queue properties to enable profiling
    auto prop_list = property_list{ property::queue::enable_profiling() };

    // create the device queue
    queue q(selector, dpc_common::exception_handler, prop_list);

    // make sure the device supports USM host allocations
    device d = q.get_device();
    if (!d.get_info<info::device::usm_host_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations\n";
      return 1;
    }

    // the golden output
    std::vector<Type> out_gold(size);

    // input and output data for the buffer version
    std::vector<Type> in_buffer(size), out_buffer(size);

    // input and output data for the restricted USM version
    // malloc_host allocates memory specifically in the host's address space
    Type* in_restricted_usm = malloc_host<Type>(size, q.get_context());
    Type* out_restricted_usm = malloc_host<Type>(size, q.get_context());
    
    // ensure that we could allocate space for both the input and output
    if (in_restricted_usm == NULL) {
      std::cerr << "ERROR: failed to allocate space for 'in_restricted_usm'\n";
      return 1;
    }
    if (out_restricted_usm == NULL) {
      std::cerr << "ERROR: failed to allocate space for 'out_restricted_usm'\n";
      return 1;
    }

    // generate some random input data and compute the golden result
    for (int i = 0; i < size; i++) {
      // generate a random value
      Type n = Type(rand() % 100);

      // populate the inputs
      in_buffer[i] = in_restricted_usm[i] = n;

      // compute the golden result
      out_gold[i] = n * i;
    }

    // run the buffer version kernel
    std::cout << "Running the buffer kernel version with size=" << size << "\n";
    std::vector<double> buffer_kernel_latency(iterations);
    for (size_t i = 0; i < iterations; i++) {
      buffer_kernel_latency[i] = BufferKernel<Type>(q, in_buffer,
                                                    out_buffer, size);
    }

    // run the restricted USM version kernel
    std::cout << "Running the restricted USM kernel version with size=" << size
              << "\n";
    std::vector<double> restricted_usm_latency(iterations);
    for (size_t i = 0; i < iterations; i++) {
      restricted_usm_latency[i] = RestrictedUSMKernel<Type>(q,
                                                            in_restricted_usm,
                                                            out_restricted_usm,
                                                            size);
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
    // validate the restricted USM version
    for (int i = 0; i < size; i++) {
      if (out_gold[i] != out_restricted_usm[i]) {
        std::cerr << "ERROR: mismatch at entry " << i
                  << " of 'RestrictedUSM' kernel output "
                  << "(" << out_gold[i] << "," << out_restricted_usm[i] << ")"
                  << "\n";
        return 1;
      }
    }

    // The FPGA emulator does not accurately represent the hardware performance
    // so we don't print performance results when running with the emulator
#ifndef FPGA_EMULATOR
    // Compute the average latency across all iterations.
    // We use the first iteration as a 'warmup' for the FPGA,
    // so we ignore its results.
    double buffer_avg_lat = std::accumulate(buffer_kernel_latency.begin() + 1,
                                            buffer_kernel_latency.end(), 0.0) /
                                        (iterations - 1);
    double restricted_usm_avg_lat =
        std::accumulate(restricted_usm_latency.begin() + 1,
                        restricted_usm_latency.end(), 0.0) /
                    (iterations - 1);

    std::cout << "Average latency for the buffer kernel: " << buffer_avg_lat
              << " ms\n";
    std::cout << "Average latency for the restricted USM kernel: "
              << restricted_usm_avg_lat << " ms\n";
#endif

    // free the allocated host usm memory
    // note that these are calls to sycl::free()
    free(in_restricted_usm, q);
    free(out_restricted_usm, q);

  } catch (exception const& e) {
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

  std::cout << "PASSED\n";

  return 0;
}
