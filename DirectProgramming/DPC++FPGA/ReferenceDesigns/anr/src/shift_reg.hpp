#ifndef __SHIFT_REG_HPP__
#define __SHIFT_REG_HPP__

#include "data_bundle.hpp"
#include "unrolled_loop.hpp"

namespace hldutils {

template <typename T, int depth>
class ShiftReg {
  T registers_[depth];

 public:
  // DO NOT Create a constructor for this; the compiler does not
  // handle it well.
  // empty default constructor since you should fill a shift-register by
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
  void Shift(T &in) {
    UnrolledLoop<0, (depth - 1)>(
        [&](int i) { registers_[i] = registers_[i + 1]; });
    registers_[depth - 1] = in;
  }

  template <int shift_amt>
  void shiftSingleVal(T &in) {
    UnrolledLoop<0, (depth - shift_amt)>(
        [&](int i) { registers_[i] = registers_[i + shift_amt]; });

    UnrolledLoop<(depth - shift_amt), depth>(
        [&](int i) { registers_[i] = in; });
  }

  template <int shift_amt>
  void ShiftMultiVals(DataBundle<T, shift_amt> &in) {
    UnrolledLoop<0, (depth - shift_amt)>(
        [&](int i) { registers_[i] = registers_[i + shift_amt]; });

    UnrolledLoop<0, shift_amt>(
        [&](int i) { registers_[(depth - shift_amt) + i] = in[i]; });
  }

  // use an accessor like this to force static accesses
  template <int idx>
  T Get() {
    // TODO: use static static asserts to check bounds of idx
    return registers_[idx];
  }

  T &operator[](int i) { return registers_[i]; }
  const T &operator[](int i) const { return registers_[i]; }
};

template <typename T, int rows, int depth>
class ShiftReg2d {
  ShiftReg<T, depth> registers_[rows];

 public:
  // DO NOT Create constructor for this; the compiler does not handle it well.
  // empty default constructor since you should fill a shift-register by
  // priming it, and if `T` is a struct, we might get a looping constructor.
  ShiftReg2d() {}

  // For a shift register with M rows and N columns, the first piece of data is
  // inserted in index [M-1][N-1], and is read out of index [0][0].
  //         j=0    1    2
  //        +----+----+----+
  // out <- | <- | <- | <- |          i=0
  //        +----+----+----+
  //        | ^- | <- | <- |          i=1
  //        +----+----+----+
  //        | ^- | <- | <- | <- input i=2
  //        +----+----+----+
  void Shift(T &in) {
    UnrolledLoop<0, (rows - 1)>([&](int i) {
      registers_[i].Shift(registers_[i + 1][0]);
    });
    registers_[(rows - 1)].Shift(in);
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
  void ShiftCol(T in[rows]) {
    UnrolledLoop<0, rows>([&](int i) {
      registers_[i].Shift(in[i]);
    });
  }

  template <int shift_amt>
  void ShiftCols(DataBundle<T, shift_amt> in[rows]) {
    UnrolledLoop<0, rows>([&](int i) {
      registers_[i].template ShiftMultiVals<shift_amt>(in[i]);
    });
  }

  // use an accessor like this to force static accesses
  template <int row, int col>
  T Get() {
    // TODO: use static static asserts to check bounds of row, col
    return registers_[row][col];
  }

  ShiftReg<T, depth> &operator[](int i) { return registers_[i]; }
  const ShiftReg<T, depth> &operator[](int i) const { return registers_[i]; }
};

}  // namespace hldutils

#endif /* __SHIFT_REG_HPP__ */
