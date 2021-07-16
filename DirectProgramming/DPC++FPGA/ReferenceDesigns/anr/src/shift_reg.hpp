// Shift Register Libary v0.2
// 2020-07-29 - v0.2 fix bug with loop bounds
// 2020-06-24 - v0.1
// author paul.white at intel dot com

#ifndef __SHIFT_REG_HPP__
#define __SHIFT_REG_HPP__

#include "data_bundle.hpp"
#include "unrolled_loop.hpp"

namespace hldutils {

template <typename T, int REG_DEPTH>
class ShiftReg {
    T registers[REG_DEPTH];

  public:
    // DO NOT Create a constructor for this; the compiler does not
    // handle it well.

    //     ShiftReg()
    //     {
    // #pragma unroll
    //       for (int i = 0; i < REG_DEPTH; i++)
    //       {
    //         registers[i] = {};
    //       }
    //     }

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
    void shift(T &in) {
        UnrolledLoop<0, (REG_DEPTH - 1)>([&](int i) {
            registers[i] = registers[i + 1];
        });
        registers[REG_DEPTH - 1] = in;
    }

    template <int SHIFT_AMT>
    void shiftSingleVal(T &in) {
        UnrolledLoop<0, (REG_DEPTH - SHIFT_AMT)>([&](int i) {
            registers[i] = registers[i + SHIFT_AMT];
        });

        UnrolledLoop<(REG_DEPTH - SHIFT_AMT), REG_DEPTH>([&](int i) {
            registers[i] = in;
        });
    }

    template <int SHIFT_AMT>
    void shiftMultiVals(DataBundle<T, SHIFT_AMT> &in) {
        UnrolledLoop<0, (REG_DEPTH - SHIFT_AMT)>([&](int i) {
            registers[i] = registers[i + SHIFT_AMT];
        });

        UnrolledLoop<0, SHIFT_AMT>([&](int i) {
            registers[(REG_DEPTH - SHIFT_AMT) + i] = in[i];
        });
    }

    // use an accessor like this to force static accesses
    template <int idx>
    T get() {
        // TODO: use static static asserts to check bounds of idx
        return registers[idx];
    }

    T &operator[](int i) { return registers[i]; }
};

template <typename T, int REG_ROWS, int REG_DEPTH>
class ShiftReg2d {
    ShiftReg<T, REG_DEPTH> registers[REG_ROWS];

  public:
    // DO NOT Create constructor for this; the compiler does not handle it well.

    //     ShiftReg2d()
    //     {
    // #pragma unroll
    //       for (int i = 0; i < REG_DEPTH; i++)
    //       {
    //         registers[i] = {};
    //       }
    //     }

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
    void shift(T &in) {
        UnrolledLoop<0, (REG_ROWS - 1)>([&](int i) {
            registers[i].shift(registers[i + 1][0]);
        });
        registers[(REG_ROWS - 1)].shift(in);
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
    void shiftCol(T in[REG_ROWS]) {
        UnrolledLoop<0, REG_ROWS>([&](int i) {
            registers[i].shift(in[i]);
        });
    }

    template <int SHIFT_AMT>
    void shiftCols(DataBundle<T, SHIFT_AMT> in[REG_ROWS]) {
        UnrolledLoop<0, REG_ROWS>([&](int i) {
            registers[i].template shiftMultiVals<SHIFT_AMT>(in[i]);
        });
    }

    // use an accessor like this to force static accesses
    template <int row, int col>
    T get() {
        // TODO: use static static asserts to check bounds of row, col
        return registers[row][col];
    }

    ShiftReg<T, REG_DEPTH> &operator[](int i) {
        return registers[i];
    }
};

} // namespace hldutils

#endif
