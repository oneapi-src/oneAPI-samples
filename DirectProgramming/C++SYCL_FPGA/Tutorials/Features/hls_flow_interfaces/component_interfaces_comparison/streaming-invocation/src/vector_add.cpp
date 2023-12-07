#include <iostream>

// oneAPI headers
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "exception_handler.hpp"

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class IDSimpleVAdd;

struct SimpleVAddKernel {
  sycl::ext::oneapi::experimental::annotated_arg<
      int *, decltype(sycl::ext::oneapi::experimental::properties{
                 sycl::ext::intel::experimental::conduit})>
      a_in;

  sycl::ext::oneapi::experimental::annotated_arg<
      int *, decltype(sycl::ext::oneapi::experimental::properties{
                 sycl::ext::intel::experimental::conduit})>
      b_in;

  sycl::ext::oneapi::experimental::annotated_arg<
      int *, decltype(sycl::ext::oneapi::experimental::properties{
                 sycl::ext::intel::experimental::conduit})>
      c_out;

  sycl::ext::oneapi::experimental::annotated_arg<
      int, decltype(sycl::ext::oneapi::experimental::properties{
               sycl::ext::intel::experimental::conduit})>
      len;

  // kernel property method to config invocation interface
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::streaming_interface<>};
  }

  void operator()() const {
    for (int idx = 0; idx < len; idx++) {
      int a_val = a_in[idx];
      int b_val = b_in[idx];
      int sum = a_val + b_val;
      c_out[idx] = sum;
    }
  }
};

constexpr int kVectorSize = 256;

int main() {
  try {
    // Use compile-time macros to select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    //  - the simulator device
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    int count = kVectorSize;  // pass array size by value

    // Create USM shared allocations in the specified buffer_location. 
    // You can also use host allocations with malloc_host(...) API
    int *a = sycl::malloc_shared<int>(count, q);
    int *b = sycl::malloc_shared<int>(count, q);
    int *c = sycl::malloc_shared<int>(count, q);
    for (int i = 0; i < count; i++) {
      a[i] = i;
      b[i] = (count - i);
    }

    std::cout << "Add two vectors of size " << count << std::endl;

    q.single_task<IDSimpleVAdd>(SimpleVAddKernel{a, b, c, count}).wait();

    // verify that VC is correct
    bool passed = true;
    for (int i = 0; i < count; i++) {
      int expected = a[i] + b[i];
      if (c[i] != expected) {
        std::cout << "idx=" << i << ": result " << c[i] << ", expected ("
                  << expected << ") A=" << a[i] << " + B=" << b[i] << std::endl;
        passed = false;
      }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cerr << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;
    std::terminate();
  }
}