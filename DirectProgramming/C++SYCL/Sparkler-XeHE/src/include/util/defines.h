/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef XeHE_DEFINES_H
#define XeHE_DEFINES_H

// Debugging help
#define XeHE_ASSERT(condition) { if(!(condition)){ std::cerr << "ASSERT FAILED: "   \
    << #condition << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; } }

// String expansion
#define _XeHE_STRINGIZE(x) #x
#define XeHE_STRINGIZE(x) _XeHE_STRINGIZE(x)

// Check that double is 64 bits
static_assert(sizeof(double) == 8, "Require sizeof(double) == 8");

// Check that int is 32 bits
static_assert(sizeof(int) == 4, "Require sizeof(int) == 4");

// Check that unsigned long long is 64 bits
static_assert(sizeof(unsigned long long) == 8, "Require sizeof(unsigned long long) == 8");


// Bounds for number of coefficient moduli
#define XeHE_COEFF_MOD_COUNT_MAX 62
#define XeHE_COEFF_MOD_COUNT_MIN 1

// Bounds for polynomial modulus degree
#define XeHE_POLY_MOD_DEGREE_MAX 32768
#define XeHE_POLY_MOD_DEGREE_MIN 2
#if 0
// Bounds for bit-length of all coefficient moduli
#define XeHE_MOD_BIT_COUNT_MAX 61
#define XeHE_MOD_BIT_COUNT_MIN 2

// Bit-length of internally used coefficient moduli, e.g., auxiliary base in BFV
#define XeHE_INTERNAL_MOD_BIT_COUNT 61

// Bounds for bit-length of user-defined coefficient moduli
#define XeHE_USER_MOD_BIT_COUNT_MAX 60
#define XeHE_USER_MOD_BIT_COUNT_MIN 2

// Bounds for the plaintext modulus
#define XeHE_PLAIN_MOD_MIN XeHE_USER_MOD_BIT_COUNT_MIN
#define XeHE_PLAIN_MOD_MAX XeHE_USER_MOD_BIT_COUNT_MAX


#if XeHE_MOD_BIT_COUNT_MAX > 32
#define XeHE_MULTIPLY_ACCUMULATE_MOD_MAX (1 << (128 - (XeHE_MOD_BIT_COUNT_MAX << 1)))
#define XeHE_MULTIPLY_ACCUMULATE_INTERNAL_MOD_MAX (1 << (128 - (XeHE_INTERNAL_MOD_BIT_COUNT << 1)))
#define XeHE_MULTIPLY_ACCUMULATE_USER_MOD_MAX (1 << (128 - (XeHE_USER_MOD_BIT_COUNT_MAX << 1)))
#else
#define XeHE_MULTIPLY_ACCUMULATE_MOD_MAX SIZE_MAX (1 << (64 - (XeHE_MOD_BIT_COUNT_MAX << 1)))
#define XeHE_MULTIPLY_ACCUMULATE_INTERNAL_MOD_MAX SIZE_MAX  (1 << (64 - ((XeHE_MOD_BIT_COUNT_MAX+1) << 1)))
#define XeHE_MULTIPLY_ACCUMULATE_USER_MOD_MAX SIZE_MAX (1 << (64 - (XeHE_USER_MOD_BIT_COUNT_MAX << 1)))
#endif
#endif
// Upper bound on the size of a ciphertext
#define XeHE_CIPHERTEXT_SIZE_MIN 2
#define XeHE_CIPHERTEXT_SIZE_MAX 16

// Detect compiler
#define XeHE_COMPILER_MSVC 1
#define XeHE_COMPILER_CLANG 2
#define XeHE_COMPILER_GCC 3

#if defined(_MSC_VER)
#define XeHE_COMPILER XeHE_COMPILER_MSVC
#elif defined(__clang__)
#define XeHE_COMPILER XeHE_COMPILER_CLANG
#elif defined(__GNUC__) && !defined(__clang__)
#define XeHE_COMPILER XeHE_COMPILER_GCC
#else
#error "Unsupported compiler"
#endif

// MSVC support
#include "util/msvc.h"

// clang support
#include "util/clang.h"

// gcc support
#include "util/gcc.h"

// Create a true/false value for indicating debug mode
#ifdef XeHE_DEBUG
#define XeHE_DEBUG_V true
#else
#define XeHE_DEBUG_V false
#endif

// Use std::byte as byte type
#ifdef XeHE_USE_STD_BYTE
#include <cstddef>
namespace xehe
{
    using XeHE_BYTE = std::byte;
}
#else
namespace xehe
{
    enum class XeHE_BYTE : unsigned char {};
}
#endif

// Use `if constexpr' from C++17
#ifdef XeHE_USE_IF_CONSTEXPR
#define XeHE_IF_CONSTEXPR if constexpr
#else
#define XeHE_IF_CONSTEXPR if
#endif

// Use [[maybe_unused]] from C++17
#ifdef XeHE_USE_MAYBE_UNUSED
#define XeHE_MAYBE_UNUSED [[maybe_unused]]
#else
#define XeHE_MAYBE_UNUSED
#endif

// Use [[nodiscard]] from C++17
#ifdef XeHE_USE_NODISCARD
#define XeHE_NODISCARD [[nodiscard]]
#else
#define XeHE_NODISCARD
#endif

// Inline Assembly definitions
#define _SIMD_WIDTH_ 16 //8//should be greater than TER_GAP_SIZE
//#define XeHE_INLINE_ASM

#endif