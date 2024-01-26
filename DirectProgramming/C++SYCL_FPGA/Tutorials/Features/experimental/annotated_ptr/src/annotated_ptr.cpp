#include <iomanip>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

using Pipe2DotProductIP =
    sycl::ext::intel::experimental::pipe<class MyPipeName1, float *>;
using Pipe2AnnotatedPtrIP =
    sycl::ext::intel::experimental::pipe<class MyPipeName2, float *>;

constexpr int kBL1 = 1;
constexpr int kBL2 = 2;

constexpr int kRows = 2;
constexpr int kCols = 5;

// The kernel 'DotProductIP' computes a weighted sum over a matrix and a vector:
// out_vec[0] = mat[0][0] * in_vec[0] + mat[0][1] * in_vec[1] + ... +
// out_vec[1] = mat[1][0] * in_vec[0] + mat[1][1] * in_vec[1] + ... +
//          ...
struct DotProductIP {
  sycl::ext::oneapi::experimental::annotated_arg<
      float *, decltype(sycl::ext::oneapi::experimental::properties{
                   sycl::ext::intel::experimental::buffer_location<kBL2>})>
      in_vec;
  sycl::ext::oneapi::experimental::annotated_arg<
      float *, decltype(sycl::ext::oneapi::experimental::properties{
                   sycl::ext::intel::experimental::buffer_location<kBL1>})>
      out_vec;

  void operator()() const {
    for (int i = 0; i < kRows; i++) {
      // read the starting pointer of the i-th row of the weight matrix
      float *p = Pipe2DotProductIP::read();

      float sum = 0.0f;
#pragma unroll kCols
      for (int j = 0; j < kCols; j++) sum += p[j] * in_vec[j];

      out_vec[i] = sum;
    }
  }
};

// The kernel 'AnnotatedPtrIP' computes the same function, but with
// `annotated_ptr` specifying the buffer location of pointers read from the host
// pipe.
struct AnnotatedPtrIP {
  sycl::ext::oneapi::experimental::annotated_arg<
      float *, decltype(sycl::ext::oneapi::experimental::properties{
                   sycl::ext::intel::experimental::buffer_location<kBL2>})>
      in_vec;
  sycl::ext::oneapi::experimental::annotated_arg<
      float *, decltype(sycl::ext::oneapi::experimental::properties{
                   sycl::ext::intel::experimental::buffer_location<kBL1>})>
      out_vec;

  void operator()() const {
    for (int i = 0; i < kRows; i++) {
      // read the starting pointer of the i-th row of the weight matrix
      float *p = Pipe2AnnotatedPtrIP::read();

      // set buffer location on p with annotated_ptr
      sycl::ext::oneapi::experimental::annotated_ptr<
          float, decltype(sycl::ext::oneapi::experimental::properties{
                     sycl::ext::intel::experimental::buffer_location<kBL1>})>
          mat{p};

      float sum = 0.0f;
#pragma unroll kCols
      for (int j = 0; j < kCols; j++) sum += mat[j] * in_vec[j];

      out_vec[i] = sum;
    }
  }
};

// verify results
bool CheckResult(float *result, float *expected, int size) {
  bool passed = true;
  for (int i = 0; i < size; i++) {
    if (result[i] != expected[i]) {
      std::cout << std::setprecision(10) << "result error! expected "
                << expected[i] << ". Received " << result[i] << "\n";
      passed = false;
    }
  }
  return passed;
}

int main() {
  bool success = true;
  try {
#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler,
                  sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // allocate memory for the flattened weight matrix. The pointers to each row
    // will be written to each kernel through a pipe.
    float *weight = sycl::malloc_shared<float>(
        kRows * kCols, q,
        sycl::ext::intel::experimental::property::usm::buffer_location(kBL1));
    assert(weight);
    for (int i = 0; i < kRows * kCols; i++) {
      weight[i] = rand() % 10;
    }

    // allocate memory and initialize for input vector
    float *input_vec = sycl::malloc_shared<float>(
        kCols, q,
        sycl::ext::intel::experimental::property::usm::buffer_location(kBL2));
    assert(input_vec);
    for (int j = 0; j < kCols; j++) input_vec[j] = rand() % 10;

    // allocate memory and initialize for output vector
    float *output_vec = sycl::malloc_shared<float>(
        kRows, q,
        sycl::ext::intel::experimental::property::usm::buffer_location(kBL1));
    assert(output_vec);
    for (int i = 0; i < kRows; i++) output_vec[i] = 0.0f;

    // Compute expected result
    float expected[kRows];
    for (int i = 0; i < kRows; i++) {
      expected[i] = 0.0f;
      for (int j = 0; j < kCols; j++) {
        expected[i] += weight[i * kCols + j] * input_vec[j];
      }
    }

    // run kernel DotProductIP
    auto event1 = q.single_task(DotProductIP{input_vec, output_vec});
    // write pointers to each row to the kernels via host pipe
    for (int i = 0; i < kRows; i++) {
      Pipe2DotProductIP::write(q, weight + i * kCols);
    }
    event1.wait();
    // verify the result
    success = CheckResult(output_vec, expected, kRows);

    // reinitialize the output vector and run kernel AnnotatedPtrIP
    for (int j = 0; j < kCols; j++) output_vec[j] = 0.0f;
    auto event2 = q.single_task(AnnotatedPtrIP{input_vec, output_vec});
    // write pointers to each row to the kernels via host pipe
    for (int i = 0; i < kRows; i++) {
      Pipe2AnnotatedPtrIP::write(q, weight + i * kCols);
    }
    event2.wait();
    // verify the result
    success &= CheckResult(output_vec, expected, kRows);

    sycl::free(input_vec, q);
    sycl::free(output_vec, q);
    sycl::free(weight, q);

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
