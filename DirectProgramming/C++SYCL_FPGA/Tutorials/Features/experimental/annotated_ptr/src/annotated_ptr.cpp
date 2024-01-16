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

#define ROWS 2
#define COLS 5

// Launch a kernel that does a weighted sum over a matrix
// out_vec[0] = mat[0][0] * in_vec[0] + mat[0][1] * in_vec[1] + ... +
// out_vec[1] = mat[1][0] * in_vec[0] + mat[1][1] * in_vec[1] + ... +
//          ...
struct pipeWithAnnotatedPtr {
  annotated_arg<float *, decltype(properties{buffer_location<kBL2>})> in_vec;
  annotated_arg<float *, decltype(properties{buffer_location<kBL1>})> out_vec;

  void operator()() const {
    for (int i = 0; i < ROWS; i++) {
      // read the starting pointer of the i-th row of the weight matrix
      float *p = MyPipe::read();

      // set buffer location on p with annotated_ptr
      annotated_ptr<float, decltype(properties{buffer_location<kBL1>})> mat{p};

      float sum = 0.0f;
#pragma unroll COLS
      for (int j = 0; j < COLS; j++)
        sum += mat[j] * in_vec[j];

      out_vec[i] = sum;
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

    // allocate memory and initialize for weight matrix
    float *weight[ROWS];
    for (int i = 0; i < ROWS; i++) {
      weight[i] =
          malloc_shared<float>(COLS, q, usm_buffer_location(kBL1));
      assert(weight[i]);
      // init data
      for (int j = 0; j < COLS; j++) {
        weight[i][j] = rand() % 10;
      }
    }
    
    // allocate memory and initialize for input vector
    auto input_vec = malloc_shared<float>(COLS, q, usm_buffer_location(kBL2));
    assert(input_vec);
    for (int j = 0; j < COLS; j++)
      input_vec[j] = rand() % 10;

    // allocate memory for output vector
    auto output_vec = malloc_shared<float>(ROWS, q, usm_buffer_location(kBL1));
    assert(output_vec); 

    // Compute expected result
    float expected[ROWS];
    for (int i = 0; i < ROWS; i++) {
      expected[i] = 0;
      for (int j = 0; j < COLS; j++) {
        expected[i] += weight[i][j] * input_vec[j];
      }
    }

    // run kernel
    auto event = q.single_task(pipeWithAnnotatedPtr{input_vec, output_vec});

    // write pointers to each row to kernel via host pipe
    for (int i = 0; i < ROWS; i++)
      MyPipe::write(q, weight[i]);

    event.wait();

    // verify results
    for (int i = 0; i < ROWS; i++) {
      if (output_vec[i] != expected[i]) {
        std::cout << std::setprecision(10)
                << "result error! expected " << expected[i] << ". Received "
                << output_vec[i] << "\n";
        success = false;
      }
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
