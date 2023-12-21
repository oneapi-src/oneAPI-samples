#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "annotated_class_util.hpp"

constexpr int kBL1 = 1;
constexpr int kAlignment = 32;
constexpr int kWidth = 256;

// Create type alias for the annotated kernel arguments, so it can be
// reused in the annotated memory allocation in the host code
// Each annotated pointer is configured with a unique `buffer_location`,
// resulting in three unique Avalon memory-mapped host interfaces.
using annotated_arg_t= sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<kBL1>,
        sycl::ext::intel::experimental::dwidth<kWidth>,
        sycl::ext::oneapi::experimental::alignment<kAlignment> })>;

struct VectorAdd {
  annotated_arg_t a;
  int size;

  void operator()() const {
    #pragma unroll 8
    for (int i = 0; i < size; i++) {
      a[i] *= 2;
    }
  }
};

bool checkResult(int *arr, int size) {
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

bool runWithUsmMalloc(sycl::queue &q) {
    // Create and initialize the host arrays
    constexpr int kN = 8;
    std::cout << "Elements in vector : " << kN << "\n";

    // Allocating host/shared memory using SYCL USM allocation API:
    // We need to specify buffer_location when allocating the host array to ensure
    // the allocated memory shares the same buffer location property as defined in the
    // corresponding kernel argument. Also since we are specifying alignment on the
    // kernel argument, we need to also specify that to the allocation call by using
    // aligned_alloc_shared API
    int *array_a = sycl::aligned_alloc_shared<int>(
        kAlignment, kN, q,
        sycl::ext::intel::experimental::property::usm::buffer_location(kBL1));

    for (int i = 0; i < kN; i++) {
      array_a[i] = i;
    }

    q.single_task(VectorAdd{array_a, kN}).wait();
    bool passed = checkResult(array_a, kN);
    free(array_a, q);
    return passed;
}

bool runWithAnnotatedAlloc(sycl::queue &q) {
    // Create and initialize the host arrays
    constexpr int kN = 8;
    std::cout << "Elements in vector : " << kN << "\n";

    // Allocating USM host/shared memory using the utility function `alloc_annotated`
    // defined in annotated_class_util.hpp: 
    // `alloc_annotated` takes the annotated_arg type defined for the kernel argument
    // as the template parameter, and returns an instance of such annotated_arg. This
    // ensures the properties of the returned memory (for example, buffer location and
    // alignment) match with the annotations on the kernel arguments.
    annotated_arg_t array_a = fpga_tools::alloc_annotated<annotated_arg_t>(kN, q);

    for (int i = 0; i < kN; i++) {
      array_a[i] = i;
    }

    q.single_task(VectorAdd{array_a, kN}).wait();
    bool passed = checkResult(array_a, kN);
    free(array_a, q);
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

    passed = runWithUsmMalloc(q);
    passed &= runWithAnnotatedAlloc(q);
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

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
