#ifndef __BYTEBITSTREAM_HPP__
#define __BYTEBITSTREAM_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#include "mp_math.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }

//
// TODO
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
  ByteBitStream() : size_(0), space_(bits), has_space_for_byte_(true) {}

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

  void PrintBuffer() {
    PRINTF("%hu: ", Size());
    for (int i = Size()-1; i >= 0; i--) {
      PRINTF("%u", buf_[i] & 0x1);
    }
    PRINTF("\n");
  }

  void PrintBinaryChar(unsigned char x) {
    for (int i = 0; i < 8; i++) {
      PRINTF("%u", (x >> (7-i)) & 0x1);
    }
  }
};

#endif // __BYTEBITSTREAM_HPP__