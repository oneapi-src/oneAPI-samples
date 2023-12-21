#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "annotated_class_util.hpp"
#include "exception_handler.hpp"

constexpr int kBL1 = 1;
constexpr int kBL2 = 2;
constexpr int kBL3 = 3;
constexpr int kAlignment = 4;

// Create type alias for the annotated kernel arguments, so it can be
// reused in the annotated memory allocation in the host code
// Each annotated pointer is configured with a unique `buffer_location`,
// resulting in three unique Avalon memory-mapped host interfaces.
using arg_a_t= sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<kBL1>,
        sycl::ext::oneapi::experimental::alignment<kAlignment> })>;
using arg_b_t = sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<kBL2>,
        sycl::ext::oneapi::experimental::alignment<kAlignment>})>;
using arg_c_t = sycl::ext::oneapi::experimental::annotated_arg<
    int *, decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<kBL3>,
        sycl::ext::oneapi::experimental::alignment<kAlignment>})>;

struct VectorAdd {
  arg_a_t a;
  arg_b_t b;
  arg_c_t c;

  int size;

  void operator()() const {
    for (int i = 0; i < size; i++) {
      c[i] = a[i] + b[i];
    }
  }
};

bool check_sum(int *a, int *b, int *c) {
  for (int i = 0; i < kN; i++) {
      if (c[i] != a[i] + b[i]) {
        std::cout << "ERROR! At index: " << i << " , expected: " << golden
                  << " , found: " << array_c[i] << "\n";
        passed = false;
      }
    }
}

bool VectorAddWithUsmMalloc(queue &q) {
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
    int *array_b = sycl::aligned_alloc_shared<int>(
        kAlignment, kN, q,
        sycl::ext::intel::experimental::property::usm::buffer_location(kBL2));
    int *array_c = sycl::aligned_alloc_shared<int>(
        kAlignment, kN, q,
        sycl::ext::intel::experimental::property::usm::buffer_location(kBL3));

    for (int i = 0; i < kN; i++) {
      array_a[i] = i;
      array_b[i] = 2 * i;
    }

    q.single_task(VectorAdd{array_a, array_b, array_c, kN}).wait();
    check_sum(array_a, array_b, array_c);

    free(array_a, q);
    free(array_b, q);
    free(array_c, q);
}

bool VectorAddWithAnnotatedAlloc(queue &q) {
    // Create and initialize the host arrays
    constexpr int kN = 8;
    std::cout << "Elements in vector : " << kN << "\n";

    // Allocating USM host/shared memory using the utility function `alloc_annotated`
    // defined in annotated_class_util.hpp: 
    // `alloc_annotated` takes the annotated_arg type defined for the kernel argument
    // as the template parameter, and returns an instance of such annotated_arg. This
    // ensures the properties of the returned memory (for example, buffer location and
    // alignment) match with the annotations on the kernel arguments.
    arg_a_t array_a = fpga_tools::alloc_annotated<arg_a_t>(kN, q);
    arg_b_t array_b = fpga_tools::alloc_annotated<arg_b_t>(kN, q);
    arg_c_t array_c = fpga_tools::alloc_annotated<arg_c_t>(kN, q);

    for (int i = 0; i < kN; i++) {
      array_a[i] = i;
      array_b[i] = 2 * i;
    }

    q.single_task(VectorAdd{array_a, array_b, array_c, kN}).wait();
    check_sum(array_a, array_b, array_c);

    free(array_a, q);
    free(array_b, q);
    free(array_c, q);
}



int main(void) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler);

    // Print out the device information.
    sycl::device device = q.get_device();
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>().c_str()
              << std::endl;

    

    bool passed = true;
    passed &= VectorAddWithUsmMalloc();
    passed &= VectorAddWithAnnotatedAlloc();
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
