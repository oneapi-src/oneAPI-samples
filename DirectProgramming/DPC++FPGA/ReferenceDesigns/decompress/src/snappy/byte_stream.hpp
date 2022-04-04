#ifndef __BYTE_STREAM_HPP__
#define __BYTE_STREAM_HPP__

// clang-format off
#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Included from DirectProgramming/DPC++FPGA/include/
#include "constexpr_math.hpp"

#include "../common/common.hpp"
// clang-format on

//
// A stream of bytes with capacity 'num_bytes'. This class allows multiple
// bytes to be read/consumed at once (0 to 'max_shift') and multiple bytes
// to be written in at once.
//
template <int num_bytes, int max_shift>
class ByteStream {
  // static asserts to check the template arguments
  static_assert(num_bytes > 0);
  static_assert(max_shift > 0);
  static_assert(max_shift <= num_bytes);

  // the number of bits used to count from 0 to 'num_bytes', inclusive
  static constexpr int count_bits = fpga_tools::Log2(num_bytes) + 1;
  using CountT = ac_uint<count_bits>;

  // the number of bits used to count from 0 to 'max_shift', inclusive
  static constexpr int shift_count_bits = fpga_tools::Log2(max_shift) + 1;
  using ShiftCountT = ac_uint<shift_count_bits>;

 public:
  ByteStream() : count_(0), space_(num_bytes) {}

  auto Count() { return count_; }
  auto Space() { return space_; }

  //
  // write in a new byte
  //
  void Write(const unsigned char& b) {
    data_[count_] = b;
    count_ += decltype(count_)(1);
    space_ -= decltype(space_)(1);
  }

  //
  // write in 'write_n' new bytes
  //
  template <size_t write_n>
  void Write(const ByteSet<write_n>& b) {
    static_assert(write_n < num_bytes);
#pragma unroll
    for (int i = 0; i < write_n; i++) {
      data_[count_ + i] = b.byte[i];
    }
    count_ += decltype(count_)(write_n);
    space_ -= decltype(space_)(write_n);
  }

  //
  // write in 'write_n' new bytes
  //
  template <size_t write_n>
  void Write(const BytePack<write_n>& b) {
    static_assert(write_n < num_bytes);
#pragma unroll
    for (int i = 0; i < write_n; i++) {
      if (i < b.valid_count) {
        data_[count_ + i] = b.byte[i];
      }
    }
    count_ += decltype(count_)(write_n);
    space_ -= decltype(space_)(write_n);
  }

  //
  // read the first element
  //
  auto Read() const { return data_[0]; }

  //
  // read the first 'read_n' elements
  template <int read_n>
  auto Read() const {
    ByteSet<read_n> ret;
#pragma unroll
    for (int i = 0; i < read_n; i++) {
      ret.byte[i] = data_[i];
    }
    return ret;
  }

  //
  // shift the stream by 1 element
  //
  void Shift() {
#pragma unroll
    for (int i = 0; i < num_bytes - 1; i++) {
      data_[i] = data_[i + 1];
    }
    count_ -= decltype(count_)(1);
    space_ += decltype(space_)(1);
  }

  //
  // shift the stream by 's' elements
  //
  template <int s>
  void Shift() {
    static_assert(s <= num_bytes);
    static_assert(s <= max_shift);

#pragma unroll
    for (int i = 0; i < num_bytes - s - 1; i++) {
      data_[i] = data_[i + s];
    }
    count_ -= decltype(count_)(s);
    space_ += decltype(space_)(s);
  }

  //
  // shift the stream by 's' elements
  //
  void Shift(ShiftCountT s) {
#pragma unroll
    for (int i = 0; i < num_bytes - 1; i++) {
      // by adding 'max_shift' extra elements to 'data_', we can avoid adding
      // the 'if (s + i < num_bytes)' condition here
      data_[i] = data_[i + s];
    }
    count_ -= decltype(count_)(s);
    space_ += decltype(space_)(s);
  }

 private:
  unsigned char data_[num_bytes + max_shift];
  CountT count_, space_;
};

#endif /* __BYTE_STREAM_HPP__ */