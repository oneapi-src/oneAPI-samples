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

SVDTestcase<float, 16, 8> testcase_16x8(
    std::vector<std::vector<float>>{
        {0.09197247, 0.14481869, 0.10407299, 0.25374938, 0.47811572, 0.63954233, 0.04104508, 0.38657333},
        {0.92316611, 0.76245709, 0.61539652, 0.89160593, 0.77919693, 0.14006746, 0.64050778, 0.88513825},
        {0.88713833, 0.21569021, 0.52698917, 0.25837260, 0.62761090, 0.16705069, 0.55006137, 0.15100562},
        {0.17933577, 0.44237509, 0.29164377, 0.04858151, 0.14284620, 0.97584930, 0.95781132, 0.97861833},
        {0.22954940, 0.53285279, 0.82211794, 0.24442794, 0.72065117, 0.82616029, 0.82302578, 0.31588218},
        {0.52760637, 0.83106858, 0.68334733, 0.58536486, 0.10177759, 0.83382267, 0.48252385, 0.33405913},
        {0.64459388, 0.44615274, 0.13607273, 0.84666874, 0.70038514, 0.05981429, 0.68471502, 0.02031992},
        {0.19211154, 0.97691734, 0.21380459, 0.18721380, 0.33669170, 0.05466270, 0.56268200, 0.05253976},
        {0.89958544, 0.17120118, 0.99595207, 0.38795272, 0.13999617, 0.22699871, 0.28511385, 0.29012966},
        {0.70594215, 0.04854467, 0.21545484, 0.15641926, 0.43467411, 0.92386666, 0.96494161, 0.19284229},
        {0.81370076, 0.90629365, 0.56153730, 0.26047083, 0.66264490, 0.83971270, 0.61051658, 0.68128875},
        {0.76390120, 0.74742154, 0.83273900, 0.83469578, 0.21863598, 0.52614912, 0.29617421, 0.87313192},
        {0.71767589, 0.42840114, 0.70372481, 0.82935507, 0.34454722, 0.92729788, 0.30406199, 0.92858277},
        {0.99486099, 0.60156528, 0.30723120, 0.68557917, 0.29556701, 0.23800143, 0.03078199, 0.19057876},
        {0.74059190, 0.68368920, 0.60495242, 0.66351287, 0.02082209, 0.90596643, 0.79826228, 0.13455221},
        {0.99834564, 0.98115456, 0.26081567, 0.80371092, 0.57020481, 0.80252733, 0.42442830, 0.54069138}},
    std::vector<float>({6.07216041, 1.71478328, 1.33994999, 1.14303961,
                        1.03316111, 0.94044096, 0.70064505, 0.65932612}));
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
    auto test_error = testcase_16x8.RunTest(q, repetitions);
    testcase_16x8.PrintResult();
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