#include <sycl/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "exception_handler.hpp"

// According to the OpenCL C spec, the format string must be in the constant
// address space. To simplify code when invoking printf, the following macros
// are defined.

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

class BasicKernel;

int main(int argc, char* argv[]) {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  queue q(selector);

  auto device = q.get_device();

  std::cout << "Running on device: "
            << device.get_info<sycl::info::device::name>().c_str()
            << std::endl;

  // Create some kernel arguments for printing.
  int x = 123;
  float y = 1.0f;
  try {
    q.submit([&](handler& h) {
       h.single_task<BasicKernel>([=]() {
         PRINTF("Result1: Hello, World!\n");
         PRINTF("Result2: %%\n");
         PRINTF("Result3: %d\n", x);
         PRINTF("Result4: %u\n", 123);
         PRINTF("Result5: %.2f\n", y);
         PRINTF("Result6: print slash_n \\n \n");
         PRINTF("Result7: Long: %ld\n", 650000L);
         PRINTF("Result8: Preceding with blanks: %10d \n", 1977);
         PRINTF("Result9: Preceding with zeros: %010d \n", 1977);
         PRINTF("Result10: Some different radices: %d %x %o %#x %#o \n", 100,
                100, 100, 100, 100);
         PRINTF("Result11: ABC%c\n", 'D');
       });
     })
        .wait();
  } catch (sycl::exception const& e) {
    std::cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    std::cout << "FAILED\n";
    std::terminate();
  }
  return 0;
}