#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#include "mp_math.hpp"

// The size of the filter can be changed at the command line
#ifndef FILTER_SIZE
#define FILTER_SIZE 9
#endif
constexpr unsigned kFilterSize = FILTER_SIZE;
static_assert(kFilterSize > 1);

// The number of pixels per cycle
#ifndef PIXELS_PER_CYCLE
#define PIXELS_PER_CYCLE 1
#endif
constexpr unsigned kPixelsPerCycle = PIXELS_PER_CYCLE;
static_assert(kPixelsPerCycle > 0);
static_assert(IsPow2(kPixelsPerCycle) > 0);

// The maximum number of columns in the image
#ifndef MAX_COLS
#define MAX_COLS 1920 // HD
//#define MAX_COLS 3840  // 4K
//#define MAX_COLS 2048
#endif
constexpr unsigned kMaxCols = MAX_COLS;
static_assert(kMaxCols > 0);
static_assert(kMaxCols > kPixelsPerCycle);

// The maximum number of rows in the image
#ifndef MAX_ROWS
#define MAX_ROWS MAX_COLS
#endif
constexpr unsigned kMaxRows = MAX_ROWS;
static_assert(kMaxRows > 0);

// pick the indexing variable size based on kMaxCols and kMaxRows
constexpr unsigned kSmallIndexTBits =
    Max(CeilLog2(kMaxCols), CeilLog2(kMaxRows));
using SmallIndexT = ac_int<kSmallIndexTBits, false>;

// add max() function to std::numeric_limits for IndexT
namespace std {
  template<> class numeric_limits<SmallIndexT> {
  public:
    static constexpr int max() { return (1 << kSmallIndexTBits) - 1; };
    static constexpr int min() { return 0; };
  };
};

// the type used for indexing the rows and columns of the image
using IndexT = short;
static_assert(std::is_integral_v<IndexT>);
static_assert(!std::is_unsigned_v<IndexT>);

// the number of bits used for the pixel
#ifndef PIXEL_BITS
#define PIXEL_BITS 8
#endif
constexpr unsigned kPixelBits = PIXEL_BITS;
static_assert(kPixelBits > 0);

// the type to use for the pixel intensity values and a temporary type
// which should have more bits than the pixel type to check for overflow.
// We will use subtraction on the temporary type, so it must be signed.
using PixelT = ac_int<kPixelBits, false>; // 'kPixelBits' bits, unsigned
using TmpT = long long;                   // 64 bits, signed
constexpr int kPixelRange = (1 << kPixelBits);
static_assert(std::is_signed_v<TmpT>);
static_assert((sizeof(TmpT) * 8) > kPixelBits);

// add min() and max() functions to std::numeric_limits for PixelT
namespace std {
  template<> class numeric_limits<PixelT> {
  public:
    static constexpr int max() { return (1 << kPixelBits) - 1; };
    static constexpr int min() { return 0; };
  };
};

// PSRN default threshold
// https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
constexpr double kPSNRDefaultThreshold = 30.0;

#endif /* __CONSTANTS_HPP__ */
