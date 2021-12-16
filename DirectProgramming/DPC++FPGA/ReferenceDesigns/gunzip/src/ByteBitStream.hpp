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

constexpr unsigned int kBufferSizeBits = 32;
constexpr unsigned int kBufferSizeBitsMask = (kBufferSizeBits - 1);
constexpr unsigned short kMaxReadBits = 15;

static_assert(fpga_tools::IsPow2(kBufferSizeBits));

//
// TODO
//
class ByteBitStream {
public:
  ByteBitStream() : widx_(0), ridx_(0), size_(0) {}

  unsigned short ReadUInt(unsigned char bits) {
    unsigned short result = 0;
    #pragma unroll
    for (unsigned char i = 0; i < kMaxReadBits; i++) {
      unsigned short the_bit =
        (buf_[(ridx_ + i) & kBufferSizeBitsMask] & 0x1);
      unsigned short val = (i < bits) ? (the_bit << i) : 0;
      result |= val;
    }

    return result;
  }

  unsigned short ReadUInt15() {
    // TODO: can we always have the next 15 bits ready to go??
    unsigned short result = 0;
    #pragma unroll
    for (unsigned char i = 0; i < 15; i++) {
      unsigned short the_bit = (buf_[(ridx_ + i) & kBufferSizeBitsMask] & 0x1);
      unsigned short val = (the_bit << i);
      result |= val;
    }

    return result;
  }

  void Shift(unsigned char bits) {
    // TODO: percompute these calculations
    ridx_ = (ridx_ + bits) & kBufferSizeBitsMask;
    size_ -= bits;
  }

  unsigned short Size() {
    return size_;
  }

  unsigned short Empty() {
    // TODO: precompute this compare
    return Size() == 0;
  }

  bool HasEnoughBits(unsigned char bits) {
    return Size() >= bits;
  }

  unsigned short Space() {
    // TODO: precompute this calculation
    return (kBufferSizeBits - Size());
  }

  bool HasSpaceForByte() {
    // TODO: precompute this compare
    return Space() >= 8; 
  }

  void NewByte(unsigned char b) {
    // put data into the buffer
    #pragma unroll
    for (unsigned short i = 0; i < 8; i++) {
      buf_[(widx_ + i) & kBufferSizeBitsMask] = ((b >> i) & 0x1);
    }

    // move the write index
    widx_ = (widx_ + 8) & kBufferSizeBitsMask;
    size_ += 8;
  }

private:
  // TODO: unsigned char here for indices and size?
  // TODO: validate unsigned char is enough bits?
  ac_int<kBufferSizeBits, false> buf_;
  unsigned short widx_, ridx_;
  short size_;

  void PrintBuffer() {
    PRINTF("%hu: ", Size());
    for (int i = Size()-1; i >= 0; i--) {
      PRINTF("%u", buf_[(ridx_ + i) & kBufferSizeBitsMask] & 0x1);
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