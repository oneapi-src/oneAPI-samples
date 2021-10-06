#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

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
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif

  queue q(device_selector);
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