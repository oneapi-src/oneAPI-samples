#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

using ValueT = int;

// offloaded computation
ValueT SomethingComplicated(ValueT val) { return (ValueT)(val * (val + 1)); }

/////////////////////////////////////////

struct FunctorRegisterMapIP {
  // Use the 'register_map' annotation on a kernel argument to specify it to be
  // a register map kernel argument.
  register_map ValueT *input;
  // Without the annotations, kernel arguments will be inferred to be register
  // map kernel arguments if the kernel invocation interface is register mapped,
  // and vise-versa.
  ValueT *output;
  // A kernel with a register map invocation interface can also independently
  // have streaming kernel arguments, when annotated by 'conduit'.
  conduit size_t n;
  register_map_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
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

  bool passed = true;

  size_t count = 16;
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

    ValueT *in = sycl::malloc_host<ValueT>(count, q);
    ValueT *functor_register_map_out = sycl::malloc_host<ValueT>(count, q);
    ValueT *golden = sycl::malloc_host<ValueT>(count, q);

    // create input and golden output data
    for (int i = 0; i < count; i++) {
      in[i] = rand() % 77;
      golden[i] = SomethingComplicated(in[i]);
      functor_register_map_out[i] = 0;
    }

    // validation lambda
    auto validate = [](auto &in, auto &out, size_t size) {
      for (int i = 0; i < size; i++) {
        if (out[i] != in[i]) {
          std::cout << "out[" << i << "] != in[" << i << "]"
                    << " (" << out[i] << " != " << in[i] << ")" << std::endl;
          return false;
        }
      }
      return true;
    };

    // Launch the kernel with a register map invocation interface implemented in
    // the functor programming model
    std::cout << "Running the kernel with a register map invocation interface "
                 "implemented in "
                 "the functor programming model"
              << std::endl;
    q.single_task(FunctorRegisterMapIP{in, functor_register_map_out, count})
        .wait();
    std::cout << "\t Done" << std::endl;

    passed &= validate(golden, functor_register_map_out, count);
    std::cout << std::endl;

    sycl::free(in, q);
    sycl::free(functor_register_map_out, q);
    sycl::free(golden, q);
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }

  if (passed) {
    std::cout << "PASSED\n";
    return 0;
  } else {
    std::cout << "FAILED\n";
    return 1;
  }
}
