#pragma once

#include "memory_system_defines.hpp"

struct SimpleNaive {
  void operator()() const {
    SimpleInputT input;
    SimpleOutputT output;

    [[intel::fpga_memory("BLOCK_RAM")]]
    int buffer[kNumRowsNaive][kNumColsNaive];

    [[intel::initiation_interval(1)]]
    for (int col = 0; col < kNumColsNaive; ++col) {
      input = InStream_SimpleNaive::read();

      #pragma unroll
      for (int row = 0; row < 4; ++row) {
        buffer[row][col] = buffer[row + 1][col];
      }
      buffer[4][col] = input;
      
      #pragma unroll
      for (int idx = 0; idx < kNumRowsNaive; ++idx) {
        output[idx] = buffer[idx][col];
      }
      
      OutStream_SimpleNaive::write(output);
    }
  }
};

struct SimpleOptimized {
  void operator()() const {
    SimpleInputT input;
    SimpleOutputT output;

    [[intel::fpga_memory("BLOCK_RAM")]]
    int buffer[kNumRowsOptimized][kNumColsOptimized];

    [[intel::initiation_interval(1)]]
    for (int row = 0; row < kNumRowsOptimized; ++row) {
      input = InStream_SimpleOptimized::read();

      #pragma unroll
      for (int col = 0; col < 4; ++col) {
        buffer[row][col] = buffer[row][col + 1];
      }
      buffer[row][4] = input;
      
      #pragma unroll
      for (int idx = 0; idx < kNumColsOptimized; ++idx) {
        output[idx] = buffer[row][idx];
      }
      
      OutStream_SimpleOptimized::write(output);
    }
  }
};
