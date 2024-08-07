#pragma once

#include "common.hpp"

// template <typename PipeIn, typename PipeOut>
// struct SimpleNaive {

//   void operator()() const {
//     [[intel::fpga_register]]
//     int input;

//     [[intel::fpga_register]]
//     std::array<int, 5> output;

//     [[intel::fpga_memory("BLOCK_RAM")]]
//     int buffer[5][500];

//     [[intel::initiation_interval(1)]]
//     for (int col = 0; col < 500; ++col) {
//       input = PipeIn::read();

//       #pragma unroll
//       for (int row = 0; row < 4; ++row) {
//         buffer[row][col] = buffer[row + 1][col];
//       }
//       buffer[4][col] = input;
//     }
//     // (TODO: remove this sum)
//     // store one at a time and load 5
//     // filling the row in a loop and access a column of 5 at once, pipe out.
//     // this way it creates 5 LD, 1 ST to one bank. creates arbitration.
//   }
// };

template <typename PipeIn, typename PipeOut>
struct SimpleNaive {

  void operator()() const {
    [[intel::fpga_register]]
    int input;

    [[intel::fpga_register]]
    std::array<int, 5> output;

    [[intel::fpga_memory("BLOCK_RAM")]]
    int buffer[5][500];

    [[intel::initiation_interval(1)]]
    for (int col = 0; col < 500; ++col) {
      input = PipeIn::read();

      #pragma unroll
      for (int row = 0; row < 4; ++row) {
        buffer[row][col] = buffer[row + 1][col];
      }
      buffer[4][col] = input;
      
      #pragma unroll
      for (int row = 0; row < 5; ++row) {
        output[row] = buffer[row][col];
      }
      
      PipeOut::write(output);
    }
  }
};

template <typename PipeIn, typename PipeOut>
struct SimpleOptimized {

  void operator()() const {
    [[intel::fpga_register]]
    int input;

    [[intel::fpga_register]]
    std::array<int, 5> output;

    [[intel::fpga_memory("BLOCK_RAM")]]
    int buffer[500][8];

    [[intel::initiation_interval(1)]]
    for (int row = 0; row < 500; ++row) {
      input = PipeIn::read();

      #pragma unroll
      for (int col = 0; col < 4; ++col) {
        buffer[row][col] = buffer[row][col + 1];
      }
      buffer[row][4] = input;
      
      #pragma unroll
      for (int idx = 0; idx < 5; ++idx) {
        output[idx] = buffer[row][idx];
      }
      
      PipeOut::write(output);
    }
  }
};
