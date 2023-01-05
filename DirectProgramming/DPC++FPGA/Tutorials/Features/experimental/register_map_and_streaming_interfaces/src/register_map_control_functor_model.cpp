#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

using ValueT = int;

// offloaded computation
ValueT SomethingComplicated(ValueT val) { return (ValueT)(val * (val + 1)); }

/////////////////////////////////////////

struct FunctorRegisterMapControlIP {
  // Use the 'register_map' annotation on a kernel argument to specify it to be
  // a register map kernel argument.
  register_map ValueT *input;
  // Without the annotations, kernel arguments will be inferred to be register
  // map kernel arguments if the kernel control interface is register map, and
  // vise-versa.
  ValueT *output;
  // A kernel with register map control can also independently have streaming
  // kernel arguments, when annotated by 'conduit'.
  conduit size_t n;
  FunctorRegisterMapControlIP(ValueT *in_, ValueT *out_, size_t N_)
      : input(in_), output(out_), n(N_) {}
  register_map_interface void operator()() const {
    for (int i = 0; i < n; i++) {
      output[i] = SomethingComplicated(input[i]);
    }
  }
};

int main(int argc, char *argv[]) {
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#elif defined(FPGA_SIMULATOR)
  sycl::ext::intel::fpga_simulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  bool passed = true;

  size_t count = 16;
  if (argc > 1)
    count = atoi(argv[1]);

  if (count <= 0) {
    std::cerr << "ERROR: 'count' must be positive" << std::endl;
    return 1;
  }

  try {
    // create the device queue
    sycl::queue q(device_selector, fpga_tools::exception_handler);

    // make sure the device supports USM host allocations
    sycl::device d = q.get_device();
    if (!d.has(sycl::aspect::usm_host_allocations)) {
      std::cerr << "ERROR: The selected device does not support USM host"
                << " allocations" << std::endl;
      return 1;
    }

    ValueT *in = sycl::malloc_host<ValueT>(count, q);
    ValueT *functorRegisterMapOut = sycl::malloc_host<ValueT>(count, q);
    ValueT *golden = sycl::malloc_host<ValueT>(count, q);

    // create input and golden output data
    for (int i = 0; i < count; i++) {
      in[i] = rand() % 77;
      golden[i] = SomethingComplicated(in[i]);
      functorRegisterMapOut[i] = 0;
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

    // Launch the kernel with register map control implemented in the functor
    // programming model
    std::cout << "Running the kernel with register map control implemented in "
                 "the functor programming model"
              << std::endl;
    q.single_task(FunctorRegisterMapControlIP{in, functorRegisterMapOut, count}).wait();
    std::cout << "\t Done" << std::endl;

    passed &= validate(golden, functorRegisterMapOut, count);
    std::cout << std::endl;

    sycl::free(in, q);
    sycl::free(functorRegisterMapOut, q);
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
