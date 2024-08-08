#include <iostream>

#include "exception_handler.hpp"

#include "memory_system_defines.hpp"
#include "simple_kernels.hpp"
#include "complex_kernels.hpp"

// Forward declaration of kernel names.
class ID_SimpleNaive;
class ID_SimpleOptimized;

class ID_BoxFilterNaive;
class ID_BoxFilterOptimized;

void random_fill(pixel_t *image, size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    image[i] = static_cast<pixel_t>(rand() % 255);
  }
}

void compute_reference_result(pixel_t *image, pixel_t *ref_result) {
  // pixel_t window[WINDOW_SZ][WINDOW_SZ];
  // for (int i = 0; i < 400; ++i) {
  //   for (int j = 0; j < 400; ++j) {
  //     if (i < 2 || j < 2) {
  //       ref_result[i][j] = 0;
  //     } else {
  //       ref_result[i][j] = box_blur_5x5(window);
  //     }
  //   }
  // }
}

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
    SimpleOutputT result;
    for (int i = 0; i < 500; ++i) {
      InStream_SimpleNaive::write(i);
      InStream_SimpleOptimized::write(i);
    }

    q.single_task<ID_SimpleNaive>(SimpleNaive{});
    q.single_task<ID_SimpleOptimized>(SimpleOptimized{});

    for (int i = 0; i < 500; ++i) {
      result = OutStream_SimpleNaive::read();
      for (auto it = result.cbegin(); it != result.cend(); ++it) {
        std::cout << *it << " ";
      }
      std::cout << std::endl;
    }
    for (int i = 0; i < 500; ++i) {
      result = OutStream_SimpleNaive::read();
      for (auto it = result.cbegin(); it != result.cend(); ++it) {
        std::cout << *it << " ";
      }
      std::cout << std::endl;
    }

    q.wait();

    // Test for complex kernels.
    random_fill(image, num_pixels);

    std::cout << "Launch kernel " << std::endl;

    q.single_task<ID_BoxFilterNaive>(BoxFilterNaive<InputPixelStream, OutputPixelStream>{num_rows, num_cols});

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