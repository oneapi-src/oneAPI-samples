#include <iostream>

// oneAPI headers
#include "exception_handler.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class vector_add_ID;

class vector_add {
  public:
    int *A_in;
    int *B_in;
    int *C_out;
    int len;

    void operator()() const {
        for (int idx = 0; idx < len; idx++) {
            int a_val  = A_in[idx];
            int b_val  = B_in[idx];
            int sum    = a_val + b_val;
            C_out[idx] = sum;
        }
    }
};

#define VECT_SIZE 256

int main() {
    bool passed = false;

    try {
// This design is tested with 2023.0, but also accounts for a syntax change in
// 2023.1
#if __INTEL_CLANG_COMPILER >= 20230100
#if FPGA_SIMULATOR
        std::cout << "using FPGA Simulator." << std::endl;
        auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
        std::cout << "using FPGA Hardware." << std::endl;
        auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
        std::cout << "using FPGA Emulator." << std::endl;
        auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
#elif __INTEL_CLANG_COMPILER >= 20230000
#if FPGA_SIMULATOR
        std::cout << "using FPGA Simulator." << std::endl;
        auto selector = sycl::ext::intel::fpga_simulator_selector{};
#elif FPGA_HARDWARE
        std::cout << "using FPGA Hardware." << std::endl;
        auto selector = sycl::ext::intel::fpga_selector{};
#else // #if FPGA_EMULATOR
        std::cout << "using FPGA Emulator." << std::endl;
        auto selector = sycl::ext::intel::fpga_emulator_selector{};
#endif
#else
        assert(false) && "this design requires oneAPI 2023.0 or 2023.1!"
#endif

        sycl::queue q(selector, fpga_tools::exception_handler,
                      sycl::property::queue::enable_profiling{});

        int count = VECT_SIZE; // pass array size by value

        // declare arrays and fill them
        // allocate in shared memory so the kernel can see them
        int *A = sycl::malloc_shared<int>(count, q);
        int *B = sycl::malloc_shared<int>(count, q);
        int *C = sycl::malloc_shared<int>(count, q);
        for (int i = 0; i < count; i++) {
            A[i] = i;
            B[i] = (count - i);
        }

        std::cout << "add two vectors of size " << count << std::endl;

        q.single_task<vector_add_ID>(vector_add{A, B, C, count}).wait();

        // verify that VC is correct
        passed = true;
        for (int i = 0; i < count; i++) {
            int expected = A[i] + B[i];
            if (C[i] != expected) {
                std::cout << "idx=" << i << ": result " << C[i] << ", expected ("
                          << expected << ") A=" << A[i] << " + B=" << B[i] << std::endl;
                passed = false;
            }
        }

        std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

        sycl::free(A, q);
        sycl::free(B, q);
        sycl::free(C, q);
    } catch (sycl::exception const &e) {
        // Catches exceptions in the host code.
        std::cerr << "Caught a SYCL host exception:\n"
                  << e.what() << "\n";

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