#include <iostream>

// oneAPI headers
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// Forward declare the kernel name in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class VectorAddID;

template <class AccA, class AccB, class AccC>
struct VectorAdd {
  public:
    AccA vec_a_in;
    AccB vec_b_in;
    AccC vec_c_out;
    int len;

    void operator()() const {
        for (int idx = 0; idx < len; idx++) {
            int a_val  = vec_a_in[idx];
            int b_val  = vec_b_in[idx];
            int sum    = a_val + b_val;
            vec_c_out[idx] = sum;
        }
    }
};

constexpr int kVectSize = 256;

int main() {
    bool passed = true;
    try {
        // Use compile-time macros to select either:
        //  - the FPGA emulator device (CPU emulation of the FPGA)
        //  - the FPGA device (a real FPGA)
        //  - the simulator device 
#if FPGA_SIMULATOR
        auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
        auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
        auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

        // create the device queue
        sycl::queue q(selector);

        auto device = q.get_device();

        std::cout << "Running on device: "
                << device.get_info<sycl::info::device::name>().c_str()
                << std::endl;

        // declare arrays and fill them
        int *vec_a = new int[kVectSize];
        int *vec_b = new int[kVectSize];
        int *vec_c = new int[kVectSize];
        for (int i = 0; i < kVectSize; i++) {
            vec_a[i] = i;
            vec_b[i] = (kVectSize - i);
        }

        std::cout << "add two vectors of size " << kVectSize << std::endl;
        {
            // copy the input arrays to buffers to share with kernel
            buffer buffer_a{vec_a, range(kVectSize)};
            buffer buffer_b{vec_b, range(kVectSize)};
            buffer buffer_c{vec_c, range(kVectSize)};

            q.submit([&](handler &h) {
                // use accessors to interact with buffers from device code
                accessor accessor_a{buffer_a, h, read_only};
                accessor accessor_b{buffer_b, h, read_only};
                accessor accessor_c{buffer_c, h, read_write, no_init};

                h.single_task<VectorAddID>(VectorAdd<decltype(accessor_a),
                                                        decltype(accessor_b),
                                                        decltype(accessor_c)>{
                accessor_a, accessor_b, accessor_c, kVectSize});
            });
        }
        // result is copied back to host automatically when accessors go out of scope.

        // verify that vec_c is correct
        for (int i = 0; i < kVectSize; i++) {
            int expected = vec_a[i] + vec_b[i];
            if (vec_c[i] != expected) {
                std::cout << "idx=" << i << ": result " << vec_c[i] << ", expected (" << expected << ") A=" << vec_a[i] << " + B=" << vec_b[i] << std::endl;
                passed = false;
            }
        }

        std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

        delete[] vec_a;
        delete[] vec_b;
        delete[] vec_c;
    } catch (sycl::exception const &e) {
        // Catches exceptions in the host code.
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
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}