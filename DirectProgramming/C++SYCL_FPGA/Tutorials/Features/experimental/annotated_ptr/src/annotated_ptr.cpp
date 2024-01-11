#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

#include <iomanip>

using namespace sycl;
using namespace ext::intel::experimental;
using namespace ext::oneapi::experimental;
using usm_buffer_location =
    ext::intel::experimental::property::usm::buffer_location;

constexpr int kBL1 = 1;
constexpr int kBL2 = 2;
using MyPipe = ext::intel::experimental::pipe<class MyPipeName, float *>;

#define ROWS 10
#define COLS 20

// Launch a kernel that does a weighted sum over a matrix
// result = data[0][0]/div[0] + data[0][1]/div[1] + ... +
//          data[1][0]/div[0] + data[1][1]/div[1] + ... +
//          ...
struct pipeWithAnnotatedPtr {
  annotated_arg<float *, decltype(properties{buffer_location<kBL1>})> result;
  annotated_arg<float *, decltype(properties{buffer_location<kBL2>})> div;

  void operator()() const {
    for (int i = 0; i < ROWS; i++) {
      float *p = MyPipe::read();

      // set buffer location on p with annotated_ptr
      annotated_arg<float *, decltype(properties{buffer_location<kBL1>})> t{p};

#pragma unroll 20
      for (int j = 0; j < COLS; j++)
        *result += t[j] / div[j];
    }
  }
};

int main() {
  bool success = true;
  try {
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    float expected = 0;

    float *testDataArray[ROWS];
    // create testData
    for (int i = 0; i < ROWS; i++) {
      testDataArray[i] =
          malloc_shared<float>(COLS, q, usm_buffer_location(kBL1));
      assert(testDataArray[i]);
    }
    // create divideData
    auto divide = malloc_shared<float>(COLS, q, usm_buffer_location(kBL2));
    assert(divide);

    // create returnData
    auto returnData = malloc_shared<float>(1, q, usm_buffer_location(kBL1));
    assert(returnData);
    *returnData = 0;

    // init data
    for (int j = 0; j < COLS; j++)
      divide[j] = rand();

    for (int i = 0; i < ROWS; i++) {
      for (int j = 0; j < COLS; j++) {
        testDataArray[i][j] = rand() * 10;
        expected += testDataArray[i][j] / divide[j];
      }
    }

    // run kernel
    auto event = q.single_task(pipeWithAnnotatedPtr{returnData, divide});

    // write pointers to each row to kernel via host pipe
    for (int i = 0; i < ROWS; i++)
      MyPipe::write(q, testDataArray[i]);

    event.wait();

    // verify results
    if (*returnData != expected) {
      std::cout << std::setprecision(10)
                << "result error! expected " << expected << ". Received "
                << *returnData << "\n";
      success = false;
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

  if (success) {
    std::cout << "PASSED: The results are correct\n";
    return 0;
  }

  return 1;
}
