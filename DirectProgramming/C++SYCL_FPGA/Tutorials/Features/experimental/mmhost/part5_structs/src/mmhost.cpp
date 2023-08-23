#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "exception_handler.hpp"

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;

using usm_buffer_location =
    sycl::ext::intel::experimental::property::usm::buffer_location;

constexpr int kBL1 = 0;
constexpr int kBL2 = 1;

using output = ac_int<6, true>;

struct input {
  int x; 
  int y; 
};

struct Struct_AC_Type_IP {

  annotated_ptr<input, decltype(properties{buffer_location<kBL1>, conduit, stable})> in;
  annotated_ptr<output, decltype(properties{buffer_location<kBL2>, conduit, stable})> out;
  int size;

  void operator()() const {
    for(int i = 0; i < size; i++){
      input temp = in.get()[i];
      out[i] = temp.x + temp.y;
    }
  }
};

int main(void) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  bool passed = true;

  try {
    // create the device queue
    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    // Print out the device information.
    sycl::device device = q.get_device();
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Create and initialize the host arrays
    constexpr int kN = 8;
    std::cout << "Elements in vector : " << kN << "\n";

    // Host array must share the same buffer location property as defined in the
    // kernel Here we may use auto* or int* when declaring the pointer interface
    input *inputObj = sycl::malloc_shared<input>(kN, q);
    output *outputObj = sycl::malloc_shared<output>(kN, q);

    for (int i = 0; i < kN; i++) {
      inputObj[i].x = i;
      inputObj[i].y = 2 * i;
    }

    q.single_task(Struct_AC_Type_IP{inputObj, outputObj, kN}).wait();
    for (int i = 0; i < kN; i++) {
      auto golden = 3 * i;
      if (outputObj[i] != golden) {
        std::cout << "ERROR! At index: " << i << " , expected: " << golden
                  << " , found: " << outputObj[i] << "\n";
        passed = false;
      }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    free(inputObj, q);
    free(outputObj, q);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
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
}