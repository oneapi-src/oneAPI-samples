#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "annotated_class_util.hpp"
#include "exception_handler.hpp"

constexpr int kBL1 = 1;
constexpr int kAlignment = 32;
constexpr int kWidth = 256;

// This type alias groups together all the properties used by the `a` pointer,
// and can be re-used in the annotated memory allocation in the host code.
// This syntax works if you have added the `-std=c++20` flag to your compiler
// command.
using annotated_arg_t = sycl::ext::oneapi::experimental::annotated_arg<
    int *, fpga_tools::properties_t<
               sycl::ext::intel::experimental::buffer_location<kBL1>,
               sycl::ext::intel::experimental::dwidth<kWidth>,
               sycl::ext::oneapi::experimental::alignment<kAlignment>>>;

struct MyIP {
  annotated_arg_t a;
  int size;

  void operator()() const {
#pragma unroll 8
    for (int i = 0; i < size; i++) {
      a[i] *= 2;
    }
  }
};

bool CheckResult(int *arr, int size) {
  bool passed = true;
  for (int i = 0; i < size; i++) {
    int golden = 2 * i;
    if (arr[i] != golden) {
      std::cout << "ERROR! At index: " << i << " , expected: " << golden
                << " , found: " << arr[i] << "\n";
      passed = false;
    }
  }
  return passed;
}

bool RunWithUsmMalloc(sycl::queue &q) {
  // Create and initialize the host arrays
  constexpr int kN = 8;
  std::cout
      << "using aligned_alloc_shared to allocate a block of shared memory\n";

  // The SYCL USM allocation API requires us to explicitly specify
  // buffer_location and alignment when allocating the host array. Unless you
  // explicitly define these as named constants in your device code (which is
  // not very tidy), you may accidentally mis-match these properties between
  // your host code and device code.
  int *array_a = sycl::aligned_alloc_shared<int>(
      kAlignment, kN, q,
      sycl::ext::intel::experimental::property::usm::buffer_location(kBL1));

  for (int i = 0; i < kN; i++) {
    array_a[i] = i;
  }

  q.single_task(MyIP{array_a, kN}).wait();
  bool passed = CheckResult(array_a, kN);
  sycl::free(array_a, q);
  return passed;
}

bool RunWithAnnotatedAlloc(sycl::queue &q) {
  // Create and initialize the host arrays
  constexpr int kN = 8;
  std::cout << "using fpga_tools::alloc_annotated to allocate a block of "
               "shared memory\n";

  // The "alloc_annotated" function extracts the buffer location and alignment
  // properties from the type alias annotated_arg_t, rather than forcing you to
  // explicitly define them in your code. This ensures the properties of the
  // returned pointer match with the annotations on the kernel arguments.
  annotated_arg_t array_a = fpga_tools::alloc_annotated<annotated_arg_t>(kN, q);

  for (int i = 0; i < kN; i++) {
    array_a[i] = i;
  }

  q.single_task(MyIP{array_a, kN}).wait();
  bool passed = CheckResult(array_a, kN);
  sycl::free(array_a, q);
  return passed;
}

int main(void) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  bool passed = false;
  try {
    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    // Print out the device information.
    sycl::device device = q.get_device();
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>().c_str()
              << std::endl;

    passed = RunWithUsmMalloc(q);
    passed &= RunWithAnnotatedAlloc(q);

    if (passed) {
      std::cout << "PASSED: all kernel results are correct\n";
    } else {
      std::cout << "FAILED\n";
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

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
