#include <assert.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "exception_handler.hpp"

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class ImplicitKernel;
class ExplicitKernel;

//
// This version of the kernel demonstrates implicit data movement
// through SYCL buffers and accessors.
//
template <typename T>
double SubmitImplicitKernel(sycl::queue &q, std::vector<T> &in,
                            std::vector<T> &out, size_t size) {
  // start the timer
  auto start = std::chrono::high_resolution_clock::now();

  {
    // set up the input and output buffers
    sycl::buffer in_buf(in);
    sycl::buffer out_buf(out);

    // launch the computation kernel
    auto kernel_event = q.submit([&](sycl::handler &h) {
      // When targeting an FPGA family/part, the compiler infers memory
      // interfaces based on the unique buffer_locations specified on kernel
      // arguments whereas when a BSP is specified to the compiler, the
      // buffer_location is used to select from the available memory interfaces
      // supported by the BSP. Here, we specify 0 on the accessor arguments
      // whereas the pointer arguments in ExplicitKernel are specified to be in
      // buffer_location 1, when targeting an FPGA family/part.
      sycl::ext::oneapi::accessor_property_list location_of_buffer{
          sycl::ext::intel::buffer_location<0>};
      sycl::accessor in_a(in_buf, h, sycl::read_only, location_of_buffer);

      sycl::ext::oneapi::accessor_property_list location_of_buffer_no_init{
          sycl::no_init, sycl::ext::intel::buffer_location<0>};
      sycl::accessor out_a(out_buf, h, sycl::write_only,
                           location_of_buffer_no_init);

      h.single_task<ImplicitKernel>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < size; i++) {
          out_a[i] = in_a[i] * i;
        }
      });
    });
  }

  // We use the scope above to synchronize the FPGA kernels.
  // Exiting the scope will cause the buffer destructors to be called
  // which will wait until the kernel finishes and copy the data back to the
  // host (if the buffer was written to).
  // Therefore, at this point in the code, we know the kernels have finished
  // and the data has been transferred back to the host.

  // stop the timer
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;

  return diff.count();
}

//
// This version of the kernel demonstrates explicit data movement
// through explicit USM.
//
template <typename T>
double SubmitExplicitKernel(sycl::queue &q, std::vector<T> &in,
                            std::vector<T> &out, size_t size) {
#if defined(IS_BSP)
  // USM device allocations are more commonly supported by FPGA boards than
  // other types of USM allocations like host and shared allocations.
  // Allocate the device memory
  T *in_ptr = sycl::malloc_device<T>(
      size, q,
      sycl::ext::intel::experimental::property::usm::buffer_location(0));
  T *out_ptr = sycl::malloc_device<T>(
      size, q,
      sycl::ext::intel::experimental::property::usm::buffer_location(0));
#else
  // When targeting an FPGA family/part, use USM host or shared allocations
  // since USM device allocations are not supported. Here we use USM shared
  // allocation.
  T *in_ptr = sycl::malloc_host<T>(
      size, q,
      sycl::ext::intel::experimental::property::usm::buffer_location(1));
  T *out_ptr = sycl::malloc_host<T>(
      size, q,
      sycl::ext::intel::experimental::property::usm::buffer_location(1));
#endif

  // ensure we successfully allocated the device memory
  if (in_ptr == nullptr) {
    std::cerr << "ERROR: failed to allocate space for 'in_ptr'\n";
    return 0;
  }
  if (out_ptr == nullptr) {
    std::cerr << "ERROR: failed to allocate space for 'out_ptr'\n";
    return 0;
  }

  // start the timer
  auto start = std::chrono::high_resolution_clock::now();

  // copy host input data to the device's memory
  auto copy_host_to_device_event =
      q.memcpy(in_ptr, in.data(), size * sizeof(T));

#if !defined(IS_BSP)
  // When targeting a FPGA family/part, the compiler infers as many global
  // memory interfaces for the design as unique buffer locations. The
  // ImplicitKernel specifies buffer_location 0 on the accessor argument
  // allowing the compiler to infer an interface for buffer_location 0.
  // Here, we use annotated_arg to specify buffer_location on the USM pointer
  // kernel argument to allow the compiler to infer an interface for
  // buffer_location 1
  sycl::ext::oneapi::experimental::annotated_arg in_ptr_d(
      in_ptr, sycl::ext::oneapi::experimental::properties{
                  sycl::ext::intel::experimental::buffer_location<1>});
  sycl::ext::oneapi::experimental::annotated_arg out_ptr_d(
      out_ptr, sycl::ext::oneapi::experimental::properties{
                   sycl::ext::intel::experimental::buffer_location<1>});
#endif

  // launch the computation kernel
  auto kernel_event = q.submit([&](sycl::handler &h) {
    // this kernel must wait until the data is copied from the host's to
    // the device's memory
    h.depends_on(copy_host_to_device_event);

    h.single_task<ExplicitKernel>([=]() [[intel::kernel_args_restrict]] {
#if defined(IS_BSP)
      // Explicitly create device pointers to inform the compiler that these
      // pointers point to device memory
      sycl::ext::intel::device_ptr<T> in_ptr_d(in_ptr);
      sycl::ext::intel::device_ptr<T> out_ptr_d(out_ptr);
#endif

      for (size_t i = 0; i < size; i++) {
        out_ptr_d[i] = in_ptr_d[i] * i;
      }
    });
  });

  // copy output data back from device to host
  auto copy_device_to_host_event = q.submit([&](sycl::handler &h) {
    // we cannot copy the output data from the device's to the host's memory
    // until the computation kernel has finished
    h.depends_on(kernel_event);
    h.memcpy(out.data(), out_ptr, size * sizeof(T));
  });

  // wait for copy back to finish
  copy_device_to_host_event.wait();

  // stop the timer
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;

  // free the device memory
  // note that these are calls to sycl::free()
  free(in_ptr, q);
  free(out_ptr, q);

  return diff.count();
}

//
// main driver program
//
int main(int argc, char *argv[]) {
  // The data type for our design. Assert that it is arithmetic.
  // Templating allows us to easily change the data type of the entire design.
  using Type = int;
  static_assert(std::is_arithmetic<Type>::value);

  // the default arguments
#if defined(FPGA_EMULATOR)
  size_t size = 10000;
  size_t iters = 1;
#elif defined(FPGA_SIMULATOR)
  size_t size = 100;
  size_t iters = 1;
#else
  size_t size = 100000000;
  size_t iters = 5;
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
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // queue properties to enable profiling
    auto prop_list =
        sycl::property_list{sycl::property::queue::enable_profiling()};

    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler, prop_list);

    // make sure the device supports USM device allocations
    auto device = q.get_device();
    if (!device.get_info<sycl::info::device::usm_device_allocations>()) {
      std::cerr << "ERROR: The selected device does not support USM device"
                << " allocations\n";
      return 1;
    }

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // input and output data
    std::vector<Type> in(size);
    std::vector<Type> out_gold(size), out_implicit(size), out_explicit(size);

    // generate some random input data
    std::generate(in.begin(), in.end(), [=] { return Type(rand() % 100); });

    // compute gold output data
    for (size_t i = 0; i < size; i++) {
      out_gold[i] = in[i] * i;
    }

    // run the ImplicitKernel
    std::cout << "Running the ImplicitKernel with size=" << size << "\n";
    std::vector<double> implicit_kernel_latency(iters);
    for (size_t i = 0; i < iters; i++) {
      implicit_kernel_latency[i] =
          SubmitImplicitKernel<Type>(q, in, out_implicit, size);
    }

    // run the ExplicitKernel
    std::cout << "Running the ExplicitKernel with size=" << size << "\n";
    std::vector<double> explicit_kernel_latency(iters);
    for (size_t i = 0; i < iters; i++) {
      explicit_kernel_latency[i] =
          SubmitExplicitKernel<Type>(q, in, out_explicit, size);
    }

    // validate the outputs
    bool passed = true;

    // validate ImplicitKernel output
    for (size_t i = 0; i < size; i++) {
      if (out_gold[i] != out_implicit[i]) {
        std::cout << "FAILED: mismatch at entry " << i
                  << " of 'ImplicitKernel' output "
                  << "(" << out_gold[i] << "," << out_implicit[i] << ")"
                  << "\n";
        passed = false;
      }
    }
    // validate ExplicitKernel kernel
    for (size_t i = 0; i < size; i++) {
      if (out_gold[i] != out_explicit[i]) {
        std::cout << "FAILED: mismatch at entry " << i
                  << " of 'ExplicitKernel' kernel output "
                  << "(" << out_gold[i] << "," << out_explicit[i] << ")"
                  << "\n";
        passed = false;
      }
    }

    if (passed) {
      // The emulator does not accurately represent real hardware performance.
      // Therefore, we don't show performance results when running in emulation.
#if !defined(FPGA_EMULATOR) && !defined(FPGA_SIMULATOR)
      double implicit_avg_lat =
          std::accumulate(implicit_kernel_latency.begin() + 1,
                          implicit_kernel_latency.end(), 0.0) /
          (double)(iters - 1);
      double explicit_avg_lat =
          std::accumulate(explicit_kernel_latency.begin() + 1,
                          explicit_kernel_latency.end(), 0.0) /
          (double)(iters - 1);

      std::cout << "Average latency for the ImplicitKernel: "
                << implicit_avg_lat << " ms\n";
      std::cout << "Average latency for the ExplicitKernel: "
                << explicit_avg_lat << " ms\n";
#endif

      std::cout << "PASSED\n";
      return 0;
    } else {
      std::cout << "FAILED\n";
      return 1;
    }

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

  return 0;
}
