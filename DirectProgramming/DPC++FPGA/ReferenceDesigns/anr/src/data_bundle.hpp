#ifndef __DATA_BUNDLE_HPP__
#define __DATA_BUNDLE_HPP__

namespace hldutils {

//
// A class used to group together 'bundle_size' elements of type 'T' into a
// struct. Similar to an array but with the copyer constructor and operator=()
// overridden to avoid expensive copys.
//
template <typename T, int bundle_size>
struct DataBundle {
  T data_[bundle_size];

  DataBundle() {}

  DataBundle(const T op) {
#pragma unroll
    for (int idx = 0; idx < bundle_size; idx++) {
      data_[idx] = op;
    }
  }

  DataBundle(const DataBundle &op) {
#pragma unroll
    for (int idx = 0; idx < bundle_size; idx++) {
      data_[idx] = op.data_[idx];
    }
  }

  DataBundle &operator=(const DataBundle &op) {
#pragma unroll
    for (int idx = 0; idx < bundle_size; idx++) {
      data_[idx] = op.data_[idx];
    }
    return *this;
  }

  bool operator==(const DataBundle &rhs) {
    bool is_equal = true;
#pragma unroll
    for (int b = 0; b < bundle_size; b++) {
      is_equal &= (data_[b] == rhs.data_[b]);
    }

    return is_equal;
  }

  // get a specific value in the bundle
  T &operator[](int i) { return data_[i]; }

  // get a raw pointer to underlying data
  T *Data() { return &data_[0]; }

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
#pragma unroll
    for (int i = 0; i < (bundle_size - 1); i++) {
      data_[i] = data_[i + 1];
    }
    data_[bundle_size - 1] = in;
  }

  template <int shift_amt>
  void ShiftSingleVal(T &in) {
#pragma unroll
    for (int i = 0; i < (bundle_size - shift_amt); i++) {
      data_[i] = data_[i + shift_amt];
    }

#pragma unroll
    for (int i = 0; i < (shift_amt); i++) {
      data_[(bundle_size - shift_amt) + i] = in;
    }
  }

  template <int shift_amt, int bundle_sz = shift_amt>
  void ShiftMultiVals(DataBundle<T, bundle_sz> &in) {
#pragma unroll
    for (int i = 0; i < (bundle_size - shift_amt); i++) {
      data_[i] = data_[i + shift_amt];
    }

#pragma unroll
    for (int i = 0; i < (shift_amt); i++) {
      data_[(bundle_size - shift_amt) + i] = in[i];
    }
  }
};

}  // namespace hldutils

#endif /* __DATA_BUNDLE_HPP__ */
