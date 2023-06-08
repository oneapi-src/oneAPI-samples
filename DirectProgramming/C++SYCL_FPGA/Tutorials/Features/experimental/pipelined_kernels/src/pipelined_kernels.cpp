#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

using ValueT = int;

// offloaded computation
ValueT SomethingComplicated(ValueT val) { return (ValueT)(val * (val + 1)); }

struct MyIP {
  conduit ValueT *input;
  streaming_pipelined_interface void operator()() const {
    ValueT temp = *input;
    *input = SomethingComplicated(temp);
  }
};

int main(int argc, char *argv[]) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  bool error = false;
  int count = 16;
  if (argc > 1) count = atoi(argv[1]);

  if (count <= 0) {
    std::cerr << "ERROR: 'count' must be positive" << std::endl;
    return 1;
  }

  try {
    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    // make sure the device supports USM host allocations
    sycl::device d = q.get_device();

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>().c_str()
              << std::endl;

    if (!d.has(sycl::aspect::usm_host_allocations)) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations" << std::endl;
      return 1;
    }

    ValueT *input_array = sycl::malloc_host<ValueT>(count, q);
    ValueT *golden = sycl::malloc_host<ValueT>(count, q);

    for (int i = 0; i < count; i++) {
      input_array[i] = rand() % 77;
      golden[i] = SomethingComplicated(input_array[i]);
    }

    std::cout << "Launching pipelined kernels consecutively" << std::endl;
    for (int i = 0; i < count; i++) {
      q.single_task(MyIP{&input_array[i]});
    }
    q.wait();
    std::cout << "\t Done" << std::endl;

    for (int i = 0; i < count; i++) {
      if (input_array[i] != golden[i]) {
        error = true;
        std::cout << "Error: Expecting " << golden[i] << " but got "
                  << input_array[i] << " from invocation" << std::endl;
      }
    }

    sycl::free(input_array, q);
    sycl::free(golden, q);
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }

  if (!error) {
    std::cout << "PASSED: Results are correct\n";
    return 0;
  } else {
    std::cout << "FAILED: Results are incorrect\n";
    return 1;
  }
}
