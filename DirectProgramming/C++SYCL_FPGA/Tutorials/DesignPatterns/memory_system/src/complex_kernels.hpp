#pragma once

#include "common.hpp"

// We want to run a 5 by 5 box blur filter on the input images.
pixel_t box_blur_5x5(const pixel_t window[WINDOW_SZ][WINDOW_SZ]) {
  pixel_t result{0};
  #pragma unroll
  for (size_t row = 0; row < WINDOW_SZ; ++row) {
    #pragma unroll
    for (size_t col = 0; col < WINDOW_SZ; ++col) {
      result += window[row][col];
    }
  }
  result /= 25;
  return result;
}


template <typename PipeIn, typename PipeOut>
struct BoxFilter
{
  const size_t num_rows;
  const size_t num_cols;

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
          
          // shift the line buffer at `col` up by one row. add new pixel at bottom.
          #pragma unroll
          for (size_t lb_row = 0; lb_row < 4; ++lb_row) {
            line_buffer[lb_row][col] = line_buffer[lb_row + 1][col];
          }
          line_buffer[4][col] = pixel_in;

          // shift the window one column to the left. new column is copied from line
          // buffer at column `col`.
          #pragma unroll
          for (size_t li = 0; li < 5; ++li) {
            #pragma unroll
            for (size_t co = 0; co < 4; ++co) {
              window[li][co] = window[li][co + 1];
            }
            window[li][4] = line_buffer[li][col];
          }
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