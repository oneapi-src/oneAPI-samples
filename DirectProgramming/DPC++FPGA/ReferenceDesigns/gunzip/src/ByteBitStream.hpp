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

//using namespace sycl;

constexpr unsigned int kBufferSizeBits = 64;
constexpr unsigned int kBufferSizeBitsMask = (kBufferSizeBits - 1);
constexpr unsigned short kMaxReadBits = 16;

static_assert(fpga_tools::IsPow2(kBufferSizeBits));

class ByteBitStream {
public:
  ByteBitStream() : widx_(0), ridx_(0), size_(0) {}

  unsigned short ReadUInt(unsigned short bits) {
    // read the bits requested
    // NOTE: assumption here is that the maximum number of bits requested
    // will be 'kMaxReadBits'
    //PrintBuffer();
    unsigned short result = 0;
    #pragma unroll
    for (unsigned short i = 0; i < kMaxReadBits; i++) {
      unsigned short the_bit =
        (buf_[(ridx_ + i) & kBufferSizeBitsMask] & 0x1);
      unsigned short val = (i < bits) ? (the_bit << i) : 0;
      result |= val;
    }

    //PRINTF("ReadUInt = %hu\n", result);

    return result;
  }

  void Shift(unsigned short bits) {
    ridx_ += bits;
    size_ -= bits;
  }

  unsigned short Size() {
    // unsigned integer overflow is defined
    // TODO: shannonize this?
    //return widx_ - ridx_;
    return size_;
  }

  unsigned short Empty() {
    return Size() == 0;
  }

  bool HasEnoughBits(unsigned short bits) {
    return Size() >= bits;
  }

  unsigned short Space() {
    // TODO: shannonize this?
    return (kBufferSizeBits - Size());
  }

  bool HasSpaceForByte() {
    return Space() >= 8; 
  }

  void NewByte(unsigned char b) {
    // put data into the buffer
    #pragma unroll
    for (unsigned short i = 0; i < 8; i++) {
      buf_[(widx_ + i) & kBufferSizeBitsMask] = ((b >> i) & 0x1);
    }

    // move the write index
    widx_ += 8;
    size_ += 8;
  }

  void SkipBits(unsigned short bits) {
    ridx_ += bits;
  }

private:
  ac_int<kBufferSizeBits, false> buf_;
  unsigned short widx_, ridx_;
  unsigned short size_;

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