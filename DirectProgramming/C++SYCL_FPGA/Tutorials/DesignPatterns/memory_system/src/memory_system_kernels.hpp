#pragma once

#include "memory_system_defines.hpp"

struct NaiveKernel {
  void operator()() const {
    SimpleInputT input;
    SimpleOutputT output;

    [[intel::fpga_memory("BLOCK_RAM")]]
    int buffer[kNumRows][kNumCols];

    [[intel::initiation_interval(1)]]
    for (int col = 0; col < kNumCols; ++col) {
      input = InStream_NaiveKernel::read();

      #pragma unroll
      for (int row = 0; row < kNumRows - 1; ++row) {
        buffer[row][col] = buffer[row + 1][col];
      }
      buffer[kNumRows - 1][col] = input;
      
      #pragma unroll
      for (int idx = 0; idx < kNumRows; ++idx) {
        output[idx] = buffer[idx][col];
      }
      
      OutStream_NaiveKernel::write(output);
    }
  }
};

struct OptimizedKernel {
  void operator()() const {
    SimpleInputT input;
    SimpleOutputT output;

    // Calculate for the number of banks, which should be a power of two.
    constexpr size_t kNumBanks = fpga_tools::Pow2(fpga_tools::CeilLog2(kNumRows));

    [[intel::fpga_memory("BLOCK_RAM")]]
    int buffer[kNumRowsOptimized][kNumBanks];

    [[intel::initiation_interval(1)]]
    for (int row = 0; row < kNumRowsOptimized; ++row) {
      input = InStream_OptimizedKernel::read();

      #pragma unroll
      for (int col = 0; col < 4; ++col) {
        buffer[row][col] = buffer[row][col + 1];
      }
      buffer[row][4] = input;
      
      #pragma unroll
      for (int idx = 0; idx < 5; ++idx) {
        output[idx] = buffer[row][idx];
      }
      
      OutStream_OptimizedKernel::write(output);
    }
  }
};
