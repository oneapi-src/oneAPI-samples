#pragma once
#include <stdint.h>

#include <array>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>

#include "matrix2d_host.hpp"
#include "pipe_matching.hpp"

namespace vvp_gray {

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

// Pixels are represented as a 16-bit integer
using PixelGray = uint16_t;

/// @brief A container for manipulating a 2D image made up of `PixelGray`. This
/// container can be used to store an image in host code.
using ImageGrey = Matrix2d<PixelGray>;

// Bundle of `PixelGray`, containing a number of parallel pixels equal to
// `kParallelPixels`.
using GrayPixelBundle = std::array<PixelGray, kParallelPixels>;

// A beat that may be transferred on a streaming interface, including sideband
// signals and a payload of `GrayPixelBundle`.
using GrayScaleBeat =
    sycl::ext::intel::experimental::StreamingBeat<GrayPixelBundle, true, true>;

template <typename T>
concept GrayScalePipe = is_pipe_of_type<T, vvp_gray::GrayScaleBeat>::value;
}  // namespace vvp_gray
