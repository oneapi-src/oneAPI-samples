//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// shift_reg.hpp

#ifndef __SHIFT_REG_HPP__
#define __SHIFT_REG_HPP__

#include <array>

#include "unrolled_loop.hpp"

namespace fpga_tools {

template <typename T, int kRegDepth>
class ShiftReg {
  T registers[kRegDepth];

 public:
  // DO NOT Create a constructor for this; the compiler does not
  // handle it well.

  //     ShiftReg()
  //     {
  // #pragma unroll
  //       for (int i = 0; i < kRegDepth; i++)
  //       {
  //         registers[i] = {};
  //       }
  //     }

  // empty default constructor since you should fill a shift register by
  // priming it, and if `T` is a struct, we might get a looping constructor.
  ShiftReg() {}

  // For a shift register with N columns, the first piece of data is inserted in
  // index [N-1], and is read out of index [0].
  //
  // ```
  //         i=0  1   2
  //        ┌───┬───┬───┐
  // out ◄─ │ r ◄─e ◄─g ◄─ input
  //        └───┴───┴───┘
  // ```
  void Shift(T in) {
    fpga_tools::UnrolledLoop<0, (kRegDepth - 1)>(
        [&](int i) { registers[i] = registers[i + 1]; });
    registers[kRegDepth - 1] = in;
  }

  template <int kShiftAmt>
  void shiftSingleVal(T in) {
    fpga_tools::UnrolledLoop<0, (kRegDepth - kShiftAmt)>(
        [&](int i) { registers[i] = registers[i + kShiftAmt]; });

    fpga_tools::UnrolledLoop<(kRegDepth - kShiftAmt), kRegDepth>(
        [&](int i) { registers[i] = in; });
  }

  template <size_t kShiftAmt>
  void ShiftMultiVals(std::array<T, kShiftAmt> in) {
    fpga_tools::UnrolledLoop<0, (kRegDepth - kShiftAmt)>(
        [&](int i) { registers[i] = registers[i + kShiftAmt]; });

    fpga_tools::UnrolledLoop<0, kShiftAmt>(
        [&](int i) { registers[(kRegDepth - kShiftAmt) + i] = in[i]; });
  }

  // use an accessor like this to force static accesses
  template <int kIdx>
  T Get() {
    // TODO: use static static asserts to check bounds of kIdx
    return registers[kIdx];
  }

  T &operator[](int i) { return registers[i]; }
};

template <typename T, int kRegRows, int kRegDepth>
class ShiftReg2d {
  ShiftReg<T, kRegDepth> registers[kRegRows];

 public:
  // DO NOT Create constructor for this; the compiler does not handle it well.

  //     ShiftReg2d()
  //     {
  // #pragma unroll
  //       for (int i = 0; i < kRegDepth; i++)
  //       {
  //         registers[i] = {};
  //       }
  //     }

  // empty default constructor since you should fill a shift register by
  // priming it, and if `T` is a struct, we might get a looping constructor.
  ShiftReg2d() {}

  // For a shift register with M rows and N columns, the first piece of data is
  // inserted in index [M-1][N-1], and is read out of index [0][0].
  //        j=  0   1   2
  //          ┌───┬───┬───┐
  //    out ◄── r ◄ e ◄ g │        i=0
  //          ├───┼───┼─▲─┤
  //          │ ┌───────┘ │
  //          │ r ◄ e ◄ g │        i=1
  //          ├───┼───┼─▲─┤
  //          │ ┌───────┘ │
  //          │ r ◄ e ◄ g ◄─ input i=2
  //          └───┴───┴───┘
  void Shift(T in) {
    fpga_tools::UnrolledLoop<0, (kRegRows - 1)>(
        [&](int i) { registers[i].Shift(registers[i + 1][0]); });
    registers[(kRegRows - 1)].Shift(in);
  }

  // For a shift register with M rows and N columns, the first column of data
  // is inserted in column [N-1], and is read out of column [0].
  //        j=0  1   2
  //       ┌───┬───┬───┐
  //      ◄─ r ◄ e ◄ g ◄─
  //       ├───┼───┼───┤
  //      ◄─ r ◄ e ◄ g ◄─
  //       ├───┼───┼───┤
  //      ◄─ r ◄ e ◄ g ◄─
  //       └───┴───┴───┘
  void shiftCol(T in[kRegRows]) {
    fpga_tools::UnrolledLoop<0, kRegRows>(
        [&](int i) { registers[i].Shift(in[i]); });
  }

  template <size_t kShiftAmt>
  void ShiftCols(std::array<T, kShiftAmt> in[kRegRows]) {
    fpga_tools::UnrolledLoop<0, kRegRows>(
        [&](int i) { registers[i].template ShiftMultiVals<kShiftAmt>(in[i]); });
  }

  // use an accessor like this to force static accesses
  template <int kRow, int kCol>
  T Get() {
    // TODO: use static static asserts to check bounds of row, col
    return registers[kRow][kCol];
  }

  ShiftReg<T, kRegDepth> &operator[](int i) { return registers[i]; }
};

}  // namespace fpga_tools

#endif