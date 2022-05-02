#ifndef __BYTE_BIT_STREAM_HPP__
#define __BYTE_BIT_STREAM_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "constexpr_math.hpp"  // included from ../../../../include

//
// A stream of bits that is filled with a byte at a time
//
// Template parameters:
//    bits: the number of bits to store in the stream. We need to store at least
//      1 byte, so this should be at least 8.
//    max_dynamic_read_bits: the maximum number of bits read from the stream
//      'dynamically' via the ReadUInt(ReadCountT) function.
//    max_shift_bits: the maximum number of bits consumed by a single call
//      to the Shift(ShiftCountT) function.
//
template <int bits, int max_dynamic_read_bits, int max_shift_bits>
class ByteBitStream {
  // static asserts to make sure the template parameters make sense
  static_assert(bits >= 8);
  static_assert(max_dynamic_read_bits > 0);
  static_assert(max_dynamic_read_bits <= bits);
  static_assert(max_shift_bits > 0);
  static_assert(max_shift_bits <= bits);

  // an ac_int to count from 0 to 'bits', inclusive
  static constexpr int count_bits = fpga_tools::Log2(bits) + 1;
  using CountT = ac_int<count_bits, false>;

  // an ac_int to count from 0 to 'max_dynamic_read_bits', inclusive
  static constexpr int dynamic_read_count_bits =
      fpga_tools::Log2(max_dynamic_read_bits) + 1;
  using ReadCountT = ac_int<dynamic_read_count_bits, false>;

  // an ac_int to count from 0 to 'max_shift_bits' inclusive
  static constexpr int shift_count_bits = fpga_tools::Log2(max_shift_bits) + 1;
  using ShiftCountT = ac_int<shift_count_bits, false>;

 public:
  ByteBitStream() : buf_(0), size_(0), space_(bits) {}

  //
  // read 'read_bits' bits from the bitstream and interpret them as an
  // unsigned int, where 'read_bits' is a runtime variable
  //
  auto ReadUInt(ReadCountT read_bits) {
    ac_int<max_dynamic_read_bits, false> mask = (1 << read_bits) - 1;
    return buf_.template slc<max_dynamic_read_bits>(0) & mask;
  }

  //
  // read 'read_bits' bits from the bitstream and interpret them as an
  // unsigned int, where 'read_bits' is constexpr
  //
  template <int read_bits>
  auto ReadUInt() {
    static_assert(read_bits <= bits);
    return buf_.template slc<read_bits>(0);
  }

  //
  // shift the bitstream by 'shift_bits' bits
  //
  void Shift(ShiftCountT shift_bits) {
    buf_ >>= shift_bits;
    size_ -= shift_bits;
    space_ += shift_bits;
  }

  //
  // shift by some number of bits to realign to a byte boundary
  //
  void AlignToByteBoundary() {
    auto bits_remaining_in_byte = size_.template slc<3>(0);
    if (bits_remaining_in_byte != 0) {
      Shift(bits_remaining_in_byte);
    }
  }

  auto Size() { return size_; }
  auto Space() { return space_; }
  bool Empty() { return size_ == 0; }
  bool HasSpaceForByte() { return space_ >= 8; }

  //
  // push in a new byte (8 bits) into the stream
  // undefined behaviour if space_ < 8
  //
  void NewByte(unsigned char b) {
    ac_int<8, false> b_ac_int(b);
    buf_.template set_slc(size_, b_ac_int);

    size_ += decltype(size_)(8);
    space_ -= decltype(space_)(8);
  }

 private:
  ac_int<bits, false> buf_;
  CountT size_, space_;
};

#endif  // __BYTE_BIT_STREAM_HPP__