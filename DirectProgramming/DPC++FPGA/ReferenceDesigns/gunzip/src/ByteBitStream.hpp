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

constexpr unsigned int kBufferSizeBits = 48;
constexpr unsigned int kBufferSizeBitsMask = (kBufferSizeBits - 1);
//static_assert(fpga_tools::IsPow2(kBufferSizeBits));

constexpr unsigned int kBufferSizeCountBits =
    fpga_tools::CeilLog2(kBufferSizeBits);
constexpr unsigned short kMaxDynamicReadBits = 5;  // see Decompressor.hpp

//
// TODO
//
class ByteBitStream {
public:
  ByteBitStream() : size_(0), space_(kBufferSizeBits),
                    has_space_for_byte_(true) {}

  auto ReadUInt(unsigned char bits) {
    ac_int<kMaxDynamicReadBits, false> tmp = 0;
    #pragma unroll
    for (unsigned char i = 0; i < kMaxDynamicReadBits; i++) {
      tmp[i] = (i < bits) ? (buf_[i] & 0x1) : 0;
    }

    return tmp;
  }

  template<int bits>
  auto ReadUIntFixed() {
    static_assert(bits <= kBufferSizeBits);
    ac_int<bits, false> tmp = 0;
    #pragma unroll
    for (unsigned char i = 0; i < bits; i++) {
      tmp[i] = buf_[i] & 0x1;
    }

    return tmp;
  }

  auto ReadUInt8() { return ReadUIntFixed<8>(); }
  auto ReadUInt15() { return ReadUIntFixed<15>(); }
  auto ReadUInt20() { return ReadUIntFixed<20>(); }
  auto ReadUInt30() { return ReadUIntFixed<30>(); }

  void Shift(unsigned char bits) {
    buf_ >>= bits & 0x1F;
    size_ -= bits & 0x1F;
    space_ += bits & 0x1F;
    has_space_for_byte_ = space_ >= 8;
  }

  auto Size() { return size_; }
  auto Space() { return space_; }
  bool Empty() { return size_ == 0; }
  bool HasSpaceForByte() { return has_space_for_byte_; }

  bool HasEnoughBits(ac_int<kBufferSizeCountBits + 1, false> bits) {
    return Size() >= bits;
  }

  void NewByte(unsigned char b) {
    // put data into the buffer
    #pragma unroll
    for (unsigned short i = 0; i < 8; i++) {
      buf_[size_ + i] = ((b >> i) & 0x1);
    }

    // move the write index
    size_ += 8;
    space_ -= 8;
    has_space_for_byte_ = space_ >= 8;
  }

private:
  ac_int<kBufferSizeBits, false> buf_;
  ac_int<kBufferSizeCountBits + 1, false> size_, space_;
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