#include <iomanip>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>
#include <sycl/ext/oneapi/annotated_arg/annotated_ptr.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

using namespace sycl;
using namespace ext::oneapi::experimental;
using usm_buffer_location =
    ext::intel::experimental::property::usm::buffer_location;

constexpr int kBL1 = 1;
constexpr int kBL2 = 2;
constexpr int kBL3 = 3;

struct PointerIP {
  int *const a;
  int *const b;
  int *const c;
  int n;

  void operator()() const {
    for (int i = 0; i < n; i++) {
      c[i] = a[i] + b[i];
    }
  }
};

struct MMHostIP {
  annotated_ptr<int, decltype(properties{buffer_location<kBL1>, awidth<32>,
                                         dwidth<32>, latency<0>,
                                         read_write_mode_readwrite, maxburst<4>,
                                         wait_request_requested})>
      a;
  annotated_ptr<int, decltype(properties{buffer_location<kBL2>, awidth<15>,
                                         dwidth<64>, latency<1>,
                                         read_write_mode_read, maxburst<1>,
                                         wait_request_not_requested})>
      b;
  annotated_ptr<int, decltype(properties{buffer_location<kBL3>, awidth<28>,
                                         dwidth<16>, latency<16>,
                                         read_write_mode_write, maxburst<1>,
                                         wait_request_not_requested})>
      c;

  int n;

  MMHostIP(int *a_, int *b_, int *c_, int n_) : a(a_), b(b_), c(c_), n(n_) {}

  void operator()() const {
    for (int i = 0; i < n; i++) {
      c[i] = a[i] + b[i];
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

  try {
    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    // Print out the device information.
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Create and initialize the host arrays
    constexpr int kN = 8;
    std::cout << "elements in vector : " << kN << "\n";

    auto host_array_A =
        malloc_shared<int>(kN, q, property_list{usm_buffer_location(kBL1)});
    assert(host_array_A);
    auto host_array_B =
        malloc_shared<int>(kN, q, property_list{usm_buffer_location(kBL2)});
    assert(host_array_B);
    auto host_array_C =
        malloc_shared<int>(kN, q, property_list{usm_buffer_location(kBL3)});
    assert(host_array_C);

    auto pointer_array_A = malloc_shared<int>(kN, q);
    assert(pointer_array_A);
    auto pointer_array_B = malloc_shared<int>(kN, q);
    assert(pointer_array_B);
    auto pointer_array_C = malloc_shared<int>(kN, q);
    assert(pointer_array_C);

    for (int i = 0; i < kN; i++) {
      host_array_A[i] = i;
      host_array_B[i] = i * 2;

      pointer_array_A[i] = i;
      pointer_array_B[i] = i * 2;
    }

    // Run the kernal code
    q.single_task(MMHostIP{host_array_A, host_array_B, host_array_C, kN})
        .wait();
    q.single_task(
         PointerIP{pointer_array_A, pointer_array_B, pointer_array_C, kN})
        .wait();

    // Check to see if results are correct
    bool passed = true;
    for (int i = 0; i < kN; i++) {
      auto golden = host_array_A[i] + host_array_B[i];
      if (host_array_C[i] != golden || pointer_array_C[i] != golden) {
        std::cout << "ERROR! At index: " << i << " , expected: " << golden
                  << " , found: " << pointer_array_C[i] << "\n";
        passed = false;
      }
    }

    if (passed) {
      std::cout << "--> PASS"
                << "\n";
    }

    // Free memory
    free(host_array_A, q);
    free(host_array_B, q);
    free(host_array_C, q);
    free(pointer_array_A, q);
    free(pointer_array_B, q);
    free(pointer_array_C, q);

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