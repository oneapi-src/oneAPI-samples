#include <iostream>

#include "exception_handler.hpp"
#include "common.hpp"
#include "simple_kernels.hpp"
#include "complex_kernels.hpp"


void random_fill(pixel_t *image, size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    image[i] = static_cast<pixel_t>(rand() % 255);
  }
}

void compute_reference_result(pixel_t *image, pixel_t *ref_result) {}

int main() {
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

    constexpr size_t num_cols = IMG_SZ;
    constexpr size_t num_rows = IMG_SZ;
    constexpr size_t num_pixels = num_rows * num_cols;
  
    pixel_t pixel;
    pixel_t image[num_pixels];
    pixel_t test_result[num_pixels];
    pixel_t reference_result[num_pixels];


    // Test for simple kernels.
    std::array<int, 5> result;
    // for (int i = 0; i < 500; ++i) {
    //   SimpleInStream::write(i);
    // }

    // q.single_task<class ID_SimpleKernel>(SimpleNaive<SimpleInStream, SimpleOutStream>{});
    // for (int i = 0; i < 500; ++i) {
    //   result = SimpleOutStream::read();
    //   for (auto it = result.cbegin(); it != result.cend(); ++it) {
    //     std::cout << *it << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // q.wait();

    for (int i = 0; i < 500; ++i) {
      SimpleInStream::write(i);
    }

    q.single_task<class ID_ComplexKernel>(SimpleOptimized<SimpleInStream, SimpleOutStream>{});
    for (int i = 0; i < 500; ++i) {
      result = SimpleOutStream::read();
      for (auto it = result.cbegin(); it != result.cend(); ++it) {
        std::cout << *it << " ";
      }
      std::cout << std::endl;
    }

    q.wait();

    // Test for complex kernels.
    random_fill(image, num_pixels);

    std::cout << "Launch kernel " << std::endl;

    q.single_task<ID_BoxFilter>(BoxFilter<InputPixelStream, OutputPixelStream>{num_rows, num_cols});

    for (size_t idx = 0; idx < num_pixels; ++idx) {
      InputPixelStream::write(q, image[idx]);
    }

    for (size_t idx = 0; idx < num_pixels; ++idx) {
      pixel = OutputPixelStream::read(q);
      test_result[idx] = pixel;
    }

    q.wait();

    compute_reference_result(image, reference_result);

    bool passed = true;
    for (size_t idx = 0; idx < num_pixels; ++idx) {
      if (test_result[idx] != reference_result[idx]) {
        passed = false;
        break;
      }
    }

    if (passed) {
      std::cout << "Verification PASSED.\n\n";
    } else {
      std::cout << "Verification FAILED.\n\n";
    }
  }
  catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::terminate();
  }
}