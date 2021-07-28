#ifndef __CONSTANTS_HPP__
#define __CONSTANTS_HPP__

// the user can disable global memory by defining the 'DISABLE_GLOBAL_MEM'
// macro: -DDISABLE_GLOBAL_MEM
#if defined(DISABLE_GLOBAL_MEM)
constexpr bool kDisableGlobalMem = true;
#else
constexpr bool kDisableGlobalMem = false;
#endif

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
//#define MAX_COLS 1920 // HD
#define MAX_COLS 3840  // 4K
#endif
constexpr unsigned kMaxCols = MAX_COLS;
static_assert(kMaxCols > 0);
static_assert(kMaxCols > kPixelsPerCycle);

// the type to use for the pixel intensity values and a temporary type
// which should have more bits than the pixel type to check for overflow.
// We will use subtraction on the temporary type, so it must be signed.
using PixelT = unsigned char;  // 8 bits, unsigned
using TmpT = long long;        // 64 bits, signed
static_assert(std::is_unsigned_v<PixelT>);
static_assert(std::is_signed_v<TmpT>);
static_assert(sizeof(TmpT) > sizeof(PixelT));

// the type used for indexing the rows and columns of the image
using IndexT = short;
static_assert(std::is_integral_v<IndexT>);
static_assert(!std::is_unsigned_v<IndexT>);

#endif /* __CONSTANTS_HPP__ */