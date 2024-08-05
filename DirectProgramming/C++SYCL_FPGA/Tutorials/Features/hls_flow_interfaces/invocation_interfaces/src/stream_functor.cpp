// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class FunctorStream;

struct Point {
  int x;
  char y;
};

/////////////////////////////////////////

struct FunctorStreamIP {
  // Annotate kernel argument with 'conduit' property
  // to specify it to be a streaming kernel argument.
  sycl::ext::oneapi::experimental::annotated_arg<
      Point, decltype(sycl::ext::oneapi::experimental::properties{
                 sycl::ext::intel::experimental::conduit})>
      input;

  // A kernel with a streaming invocation interface can also independently
  // have register-mapped kernel arguments, when annotated by 'register_map'
  // property.
  sycl::ext::oneapi::experimental::annotated_arg<
      Point *, decltype(sycl::ext::oneapi::experimental::properties{
                   sycl::ext::intel::experimental::register_map})>
      output;

  // Without the annotation, kernel argument will be inferred to be streaming
  // kernel arguments if the kernel invocation interface is streaming, and
  // vice-versa.
  int n;

  // Kernel properties method to configure the kernel to be a kernel with
  // streaming invocation interface.
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::
            streaming_interface_accept_downstream_stall};
  }

  void operator()() const {
    // For annotated_arg of struct type, explicitly cast away the annotated_arg
    // to prevent compiler error.
    struct Point ret;
    ret.x = 0;
    ret.y = ((Point)input).y;

    for (int i = 0; i < n; i++) {
      ret.x += ((Point)input).x;
      ret.y += 1;
    }
    *output = ret;
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

    Point input;
    input.x = 1;
    input.y = 'a';

    Point *functor_streaming_out = sycl::malloc_host<Point>(count, q);
    Point *golden_out = sycl::malloc_host<Point>(count, q);

    // test that mallocs did not return nullptr
    assert(functor_streaming_out);
    assert(golden_out);

    // Compute golden output data
    Point ret;
    ret.x = 0;
    ret.y = input.y;

    for (int i = 0; i < (count); i++) {
      ret.x += input.x;
      ret.y += 1;
    }
    *golden_out = ret;

    // validation lambda
    auto validate = [](auto *golden_out, auto *functor_streaming_out) {
      if (functor_streaming_out->x != golden_out->x ||
          functor_streaming_out->y != golden_out->y) {
        std::cout << "Expected: \n";
        std::cout << "functor_streaming_out->x = " << golden_out->x << "\n";
        std::cout << "functor_streaming_out->y = " << golden_out->y << "\n";
        std::cout << "Got: \n";
        std::cout << "functor_streaming_out->x = " << functor_streaming_out->x
                  << "\n";
        std::cout << "functor_streaming_out->y = " << functor_streaming_out->y
                  << "\n";
        std::cout << "FAILED\n";
        return false;
      }
      return true;
    };

    // Launch the kernel with a streaming invocation interface implemented in
    // the functor programming model
    std::cout << "Running the kernel with streaming invocation interface "
                 "implemented in the "
                 "functor programming model"
              << std::endl;
    q.single_task<FunctorStream>(
         FunctorStreamIP{input, functor_streaming_out, count})
        .wait();
    std::cout << "\t Done" << std::endl;

    passed &= validate(golden_out, functor_streaming_out);
    std::cout << std::endl;

    sycl::free(functor_streaming_out, q);
    sycl::free(golden_out, q);
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
