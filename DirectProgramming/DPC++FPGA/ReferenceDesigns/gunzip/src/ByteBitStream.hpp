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
  ByteBitStream() : widx_(0), ridx_(0), size_(0), space_(kBufferSizeBits) {}

  unsigned short ReadUInt(unsigned char bits) {
    ac_int<kMaxReadBits, false> tmp = 0;
    #pragma unroll
    for (unsigned char i = 0; i < kMaxReadBits; i++) {
      tmp[i] = (i < bits) ? (buf_[(ridx_ + i) & kBufferSizeBitsMask] & 0x1) : 0;
    }

    return (unsigned short)tmp;
  }

  unsigned short ReadUInt15() {
    ac_int<15, false> tmp = 0;
    #pragma unroll
    for (unsigned char i = 0; i < 15; i++) {
      tmp[i] = buf_[(ridx_ + i) & kBufferSizeBitsMask] & 0x1;
    }

    return (unsigned short)tmp;
  }

  ac_int<20, false> ReadUInt20() {
    ac_int<20, false> tmp = 0;
    #pragma unroll
    for (unsigned char i = 0; i < 20; i++) {
      tmp[i] = buf_[(ridx_ + i) & kBufferSizeBitsMask] & 0x1;
    }
    return tmp;
  }

  void Shift(unsigned char bits) {
    // TODO: percompute these calculations
    ridx_ = (ridx_ + bits) & kBufferSizeBitsMask;
    size_ -= bits;
    space_ += bits;
  }

  unsigned short Size() {
    return size_;
  }

  unsigned short Space() {
    return space_;
  }

  bool Empty() {
    // TODO: precompute
    return size_ == 0;
  }

  bool HasEnoughBits(unsigned char bits) {
    return Size() >= bits;
  }

  bool HasSpaceForByte() {
    // TODO: precompute
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
    space_ -= 8;
  }

private:
  ac_int<kBufferSizeBits, false> buf_;

  // TODO: use ac_int here for exact number of bits needed?
  unsigned char widx_, ridx_;
  unsigned char size_, space_;
  static_assert(std::numeric_limits<unsigned char>::max() > kBufferSizeBits);

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