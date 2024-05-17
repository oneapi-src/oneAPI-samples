#pragma once
#include <stdint.h>

#include <array>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

namespace conv2d {

#ifndef PIXEL_BITS
#warning "PIXEL_BITS undefined. choosing PIXEL_BITS=10
#define PIXEL_BITS 10
#endif

// Number of bits-per-color-channel that this IP will expect. Define the
// `PIXEL_BITS` macro to override this at compile-time.
constexpr uint32_t kBitsPerChannel = PIXEL_BITS;

#ifndef PARALLEL_PIXELS
#warning "PARALLEL_PIXELS undefined. choosing PARALLEL_PIXELS=1
#define PARALLEL_PIXELS 1
#endif

// Number of pixels-in-parallel that this IP will expect. Define the
// `PARALLEL_PIXELS` macro to override this at compile-time.
constexpr uint32_t kParallelPixels = PARALLEL_PIXELS;

#ifndef WINDOW_SZ
#warning "WINDOW_SZ undefined. Choosing WINDOW_SZ=3
#define WINDOW_SZ 3
#endif
// kernel size to use for sliding window
constexpr uint32_t kWindowSize = WINDOW_SZ;

#ifndef MAX_COLS
#warning "MAX_COLS undefined. Choosing MAX_COLS=4096
#define MAX_COLS 4096
#endif
// kernel size to use for sliding window
constexpr uint32_t kMaxCols = MAX_COLS;

#pragma pack(push, 1)
struct PixelRGB {
  // no constructor as this results in additional loops that kill performance

  // VVP expects the red channel in the most significant bits, so put it last
  // in the struct
  uint16_t b;  // ac_int<kBitsPerChannel, false> hurts debugability.
  uint16_t g;  // ac_int<kBitsPerChannel, false> hurts debugability.
  uint16_t r;  // ac_int<kBitsPerChannel, false> hurts debugability.
};
#pragma pack(pop)

// Pixels are represented as a 16-bit integer
using PixelType = uint16_t;

// Bundle of `PixelType`, containing a number of parallel pixels equal to
// `kParallelPixels`.
using GreyPixelBundle = std::array<PixelType, kParallelPixels>;

// Bundle of `PixelRGB`, containing a number of parallel pixels equal to
// `kParallelPixels`.
using RGBPixelBundle = std::array<PixelRGB, kParallelPixels>;

// A beat that may be transferred on a streaming interface, including sideband
// signals and a payload of `GreyPixelBundle`.
using GreyScaleBeat =
    sycl::ext::intel::experimental::StreamingBeat<GreyPixelBundle, true, true>;

// A beat that may be transferred on a streaming interface, including sideband
// signals and a payload of `RGBPixelBundle`.
using RGBBeat =
    sycl::ext::intel::experimental::StreamingBeat<RGBPixelBundle, true, true>;
}  // namespace conv2d
