#pragma once

#include "memory_system_defines.hpp"

struct NaiveKernel {
  void operator()() const {
    [[intel::fpga_memory("BLOCK_RAM")]]
    int array2d[kNumRows][kNumCols];

    [[intel::initiation_interval(1)]]
    for (int col = 0; col < kNumCols; ++col) {
      SimpleInputT input = InStreamNaiveKernel::read();
      SimpleOutputT output;

      #pragma unroll
      for (int row = 0; row < kNumRows - 1; ++row) {
        array2d[row][col] = array2d[row + 1][col];
      }
      array2d[kNumRows - 1][col] = input;
      
      #pragma unroll
      for (int row = 0; row < kNumRows; ++row) {
        output[row] = array2d[row][col];
      }
      
      OutStreamNaiveKernel::write(output);
    }
  }
};

struct OptimizedKernel {
  void operator()() const {
    // Calculate for the number of banks, which should be a power of two.
    constexpr size_t kNumBanks = fpga_tools::Pow2(fpga_tools::CeilLog2(kNumRows));

    [[intel::fpga_memory("BLOCK_RAM")]]
    int array2d[kNumRowsOptimized][kNumBanks];

    [[intel::initiation_interval(1)]]
    for (int row = 0; row < kNumRowsOptimized; ++row) {
      SimpleInputT input = InStreamOptKernel::read();
      SimpleOutputT output;

      #pragma unroll
      for (int col = 0; col < 4; ++col) {
        array2d[row][col] = array2d[row][col + 1];
      }
      array2d[row][4] = input;
      
      #pragma unroll
      for (int col = 0; col < 5; ++col) {
        output[col] = array2d[row][col];
      }
      
      OutStreamOptKernel::write(output);
    }
  }
};
