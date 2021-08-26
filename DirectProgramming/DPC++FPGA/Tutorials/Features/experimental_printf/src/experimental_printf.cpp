#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "dpc_common.hpp"

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

  try {
    q.submit([&](handler& h) {
       h.single_task<BasicKernel>([=
       ]() [[intel::kernel_args_restrict]] {  // NO-FORMAT: Attribute
         PRINTF("PASS Result1: Hello, World!\n");
         PRINTF("PASS Result2: %%\n");
         PRINTF("PASS Result3: %d\n", 123);
         PRINTF("PASS Result4: %u\n", 123);
         PRINTF("PASS Result5: %.2f\n", 1.0f);
         PRINTF("PASS Result6: print slash_n \\n \n");
         PRINTF("PASS Result7: Long: %ld\n", 650000L);
         PRINTF("PASS Result8: Preceding with blanks: %10d \n", 1977);
         PRINTF("PASS Result9: Preceding with zeros: %010d \n", 1977);
         PRINTF("PASS Resulta: Some different radices: %d %x %o %#x %#o \n",
                100, 100, 100, 100, 100);
         PRINTF("PASS Resultb: ABC%c\n", 'D');
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