#ifndef __BYTEBITSTREAM_HPP__
#define __BYTEBITSTREAM_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "mp_math.hpp"

//
// A stream of bits that is filled with a byte at a time
//
template<int bits, int max_dynamic_read_bits, int max_shift_bits>
class ByteBitStream {
  static_assert(bits > 0);
  static_assert(max_dynamic_read_bits > 0);
  static_assert(max_dynamic_read_bits <= bits);
  static_assert(max_shift_bits > 0);
  static_assert(max_shift_bits <= bits);

  using BufferT = ac_int<bits, false>;
  static constexpr int count_bits = fpga_tools::Log2(bits) + 1;
  using CountT = ac_int<count_bits, false>;
  static constexpr int dynamic_read_count_bits =
      fpga_tools::Log2(max_dynamic_read_bits) + 1;
  using ReadCountT = ac_int<dynamic_read_count_bits, false>;
  static constexpr int shift_count_bits = fpga_tools::Log2(max_shift_bits) + 1;
  using ShiftCountT = ac_int<shift_count_bits, false>;

public:
  ByteBitStream() : buf_(0), size_(0), space_(bits),
                    has_space_for_byte_(true) {}

  auto ReadUInt(ReadCountT read_bits) {
    ac_int<max_dynamic_read_bits, false> mask = (1 << read_bits) - 1;
    return buf_.template slc<max_dynamic_read_bits>(0) & mask;
  }

  template<int read_bits>
  auto ReadUInt() {
    static_assert(read_bits <= bits);
    return buf_.template slc<read_bits>(0);
  }

  void Shift(ShiftCountT shift_bits) {
    buf_ >>= shift_bits;
    size_ -= shift_bits;
    space_ += shift_bits;
    has_space_for_byte_ = space_ >= 8;
  }

  void AlignToByteBoundary() {
    ac_int<3, false> pos_in_byte = size_.template slc<3>(0);
    if (pos_in_byte != 0) {
      ac_int<3, false> bits_to_drop = ac_int<3, false>(7) - pos_in_byte;
      Shift(bits_to_drop);
    }
  }

  auto Size() { return size_; }
  auto Space() { return space_; }
  bool Empty() { return size_ == 0; }
  bool HasSpaceForByte() { return has_space_for_byte_; }

  void NewByte(unsigned char b) {
    // put data into the buffer
    ac_int<8, false> b_ac_int(b);
    buf_.template set_slc(size_, b_ac_int);

    size_ += decltype(size_)(8);
    has_space_for_byte_ = space_ >= 16;
    space_ -= decltype(space_)(8);
  }

private:
  BufferT buf_;
  CountT size_, space_;
  bool has_space_for_byte_;
};

#endif // __BYTEBITSTREAM_HPP__