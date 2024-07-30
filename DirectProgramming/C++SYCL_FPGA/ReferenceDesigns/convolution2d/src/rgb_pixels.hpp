#pragma once
#include <stdint.h>

#include <array>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

#include "matrix2d_host.hpp"
#include "pipe_matching.hpp"

namespace vvp_rgb {

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

/// @brief A container for manipulating a 2D image made up of `PixelRGB`. This
/// container can be used to store an image in host code.
using ImageRGB = Matrix2d<PixelRGB>;

// Bundle of `PixelRGB`, containing a number of parallel pixels equal to
// `kParallelPixels`.
using RGBPixelBundle = std::array<PixelRGB, kParallelPixels>;

// A beat that may be transferred on a streaming interface, including sideband
// signals and a payload of `RGBPixelBundle`.
using RGBBeat =
    sycl::ext::intel::experimental::StreamingBeat<RGBPixelBundle, true, true>;

template <typename T>
concept RGBPipe = is_pipe_of_type<T, vvp_rgb::RGBBeat>::value;

}  // namespace vvp_rgb
