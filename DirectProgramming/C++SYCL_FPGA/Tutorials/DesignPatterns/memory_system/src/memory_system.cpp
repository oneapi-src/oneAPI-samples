#include <iostream>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

#include "constexpr_math.hpp"
#include "exception_handler.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;

// size of the line buffer that holds the image pixels.
#define LB_SZ 1024

// size of the filter window.
#define WINDOW_SZ 5

// input image size.
#define IMG_SZ 400

using pixel_t = unsigned int;

// Definition of input and output pipes.
using PixelPipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::bits_per_symbol<8>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>));

class ID_InStream;
using InputPixelStream =
    sycl::ext::intel::experimental::pipe<ID_InStream, pixel_t, 0,
                                         PixelPipePropertiesT>;

class ID_OutStream;
using OutputPixelStream =
    sycl::ext::intel::experimental::pipe<ID_OutStream, pixel_t, 0,
                                         PixelPipePropertiesT>;

class ID_BoxFilter;
using ConduitArgT = decltype(sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::conduit});

// We want to run a 5 by 5 box blur filter on the input images.
pixel_t box_blur_5x5(const pixel_t window[WINDOW_SZ][WINDOW_SZ]) {
  pixel_t result{0};
  fpga_tools::UnrolledLoop<0, 5>([&](auto row)
                                 { fpga_tools::UnrolledLoop<0, 5>([&](auto col)
                                                                  { result += window[row][col]; }); });
  result /= 25;
  return result;
}

template <typename PipeIn, typename PipeOut>
struct BoxFilter
{
  sycl::ext::oneapi::experimental::annotated_arg<size_t, ConduitArgT> num_rows;
  sycl::ext::oneapi::experimental::annotated_arg<size_t, ConduitArgT> num_cols;

  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::streaming_interface<>};
  }

  void operator()() const {
    [[intel::fpga_register]]
    pixel_t pixel_in;

    [[intel::fpga_register]]
    pixel_t pixel_out;

    [[intel::fpga_register]]
    pixel_t pixel_computed;

    // Implement line buffer on FPGA on-chip memory.
#ifndef OPTIMIZED_EXAMPLE
    [[intel::fpga_memory("BLOCK_RAM")]]
    pixel_t line_buffer[WINDOW_SZ][LB_SZ];
#else
    constexpr size_t kNumBanks = fpga_tools::Pow2(fpga_tools::CeilLog2(WINDOW_SZ));
    [[intel::fpga_memory("BLOCK_RAM")]]
    pixel_t line_buffer[LB_SZ][kNumBanks];
#endif

    // Pixels to compute filter with.
    [[intel::fpga_register]]
    pixel_t window[WINDOW_SZ][WINDOW_SZ];

    [[intel::initiation_interval(1)]]
    for (int row = 0; (row < LB_SZ) && row < num_rows + 2; row++) {
      [[intel::initiation_interval(1)]]
      for (int col = 0; (col < LB_SZ) && (col < num_cols + 2); col++) {

        if (row < num_rows && col < num_cols) {
          pixel_in = PipeIn::read();

          fpga_tools::UnrolledLoop<0, 4>([&](auto l)
                                         { line_buffer[l][col] = line_buffer[l + 1][col]; });
          line_buffer[4][col] = pixel_in;

          fpga_tools::UnrolledLoop<0, 5>([&](auto li)
                                         {
            fpga_tools::UnrolledLoop<0, 4>([&](auto co) {
              window[li][co] = window[li][co + 1];
            });
            window[li][4] = line_buffer[li][col]; });
        }
        pixel_computed = box_blur_5x5(window);

        if ((row >= 2) && (col >= 2)) {
          pixel_out = 0;

          if (((row >= 4) && (row < num_rows) && (col >= 4) && (col < num_cols))) {
            pixel_out = pixel_computed;
          }
          PipeOut::write(pixel_out);
        }
      }
    }
  }
};

void random_fill(pixel_t *image, size_t sz) {
  for (size_t i = 0; i < sz; ++i) {
    image[i] = static_cast<pixel_t>(rand() % 255);
  }
}

void compute_reference_result(pixel_t *image, pixel_t *ref_result) {
  
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