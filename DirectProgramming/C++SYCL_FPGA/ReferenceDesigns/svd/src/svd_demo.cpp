#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "exception_handler.hpp"
#include "svd.hpp"
#include "svd_helper.hpp"
#include "svd_testcase.hpp"

int main(int argc, char *argv[]) {
#if FPGA_HARDWARE
  int repetitions = 16384;
#else
  int repetitions = 1;
#endif
  if (argc == 2) {
    repetitions = std::stoi(argv[1]);
  }

  try {
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // Enable the queue profiling to time the execution
    sycl::property_list queue_properties{
        sycl::property::queue::enable_profiling()};
    sycl::queue q =
        sycl::queue(selector, fpga_tools::exception_handler, queue_properties);
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

#if FPGA_SIMULATOR
    auto test_error = small_4x4.run_test(q, repetitions);
    small_4x4_trivial.print_result();
#else
    auto test_error = testcase_16x8.run_test(q, repetitions);
    testcase_16x8.print_result();
#endif
    bool passed = test_error < 0.1;
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }
}