#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

using namespace sycl;
using namespace ext::oneapi::experimental;

using usm_buffer_location =
    ext::intel::experimental::property::usm::buffer_location;

constexpr int kBL1 = 1;
constexpr int kBL2 = 2;
constexpr int kBL3 = 3;

struct MultiMMIP {

  //Declare the pointer interfaces to be used in this kernel,
  //look at the other kernals to compare the difference 
  annotated_ptr<int, decltype(properties{
    buffer_location<kBL1>,
    awidth<32>, 
    dwidth<32>, 
    latency<0>, 
    read_write_mode_read,
    maxburst<4>
  })> x;

  annotated_ptr<int, decltype(properties{
    buffer_location<kBL2>,
    awidth<32>, 
    dwidth<32>, 
    latency<0>, 
    read_write_mode_read,
    maxburst<4>
  })> y;

  annotated_ptr<int, decltype(properties{
    buffer_location<kBL3>,
    awidth<32>, 
    dwidth<32>, 
    latency<0>, 
    read_write_mode_write,
    maxburst<4>
  })> z;

  int size;

  void operator()() const {

    #pragma unroll 4
    for(int i = 0; i < size; i++){
      z[i] = x[i] + y[i];
    }
  }
};

int main(void){
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

    // Host array must share the same buffer location property as defined in the kernel
    // Here we may use auto* or int* when declaring the pointer interface
    auto *array_A = malloc_shared<int>(kN, q, property_list{usm_buffer_location(kBL1)});
    auto *array_B = malloc_shared<int>(kN, q, property_list{usm_buffer_location(kBL2)});
    int *array_C = malloc_shared<int>(kN, q, property_list{usm_buffer_location(kBL3)});

    for(int i = 0; i < kN; i++){
        array_A[i] = i;
        array_B[i] = 2*i;
    }

    q.single_task(MultiMMIP{array_A, array_B, array_C, kN}).wait();
    for (int i = 0; i < kN; i++) {
      auto golden = 3*i;
      if (array_C[i] != golden) {
        std::cout << "ERROR! At index: " << i << " , expected: " << golden
                  << " , found: " << array_C[i] << "\n";
        passed = false;
      }
    }

    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

    free(array_A, q);
    free(array_B, q);
    free(array_C, q);

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  catch(sycl::exception const &e){
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