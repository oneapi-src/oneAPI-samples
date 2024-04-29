#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <vector>

#include "exception_handler.hpp"
#include "svd.hpp"
#include "svd_testcase.hpp"

// clang-format off
SVDTestcase<float, 4, 4> small_4x4(
    std::vector<std::vector<float>>{
        {0.47084338, 0.99594452, 0.47982739, 0.69202168},
        {0.45148837, 0.72836647, 0.64691844, 0.62442883},
        {0.80974833, 0.82555856, 0.30709051, 0.58230306},
        {0.97898197, 0.98520343, 0.40133633, 0.85319924}},
    std::vector<float>({2.79495619, 0.44521050, 0.19458290, 0.07948970}));
// clang-format on

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
    auto test_error = small_4x4.RunTest(q, repetitions);
    small_4x4.PrintResult();
#else
    SVDTestcase<float, 32, 32> large_testcase;
    auto test_error = large_testcase.RunTest(q, repetitions);
    large_testcase.PrintResult();
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