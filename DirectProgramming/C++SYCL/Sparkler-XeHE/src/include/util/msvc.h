/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef XeHE_MSVC_H
#define XeHE_MSVC_H

#if XeHE_COMPILER == XeHE_COMPILER_MSVC

// Require Visual Studio 2017 version 15.3 or newer
#if (_MSC_VER < 1911)
#error "Microsoft Visual Studio 2017 version 15.3 or newer required"
#endif

// Read in config.h
#include "util/config.h"

// Do not throw when Evaluator produces transparent ciphertexts
//#undef XeHE_THROW_ON_TRANSPARENT_CIPHERTEXT

// In Visual Studio redefine std::byte (XeHE_BYTE)
#undef XeHE_USE_STD_BYTE

// In Visual Studio for now we disable the use of std::shared_mutex
#undef XeHE_USE_SHARED_MUTEX

// Are we compiling with C++17 or newer
#if (__cplusplus >= 201703L)

// Use `if constexpr'
#define XeHE_USE_IF_CONSTEXPR

// Use [[maybe_unused]]
#define XeHE_USE_MAYBE_UNUSED

// Use [[nodiscard]]
#define XeHE_USE_NODISCARD

#else
#undef XeHE_USE_IF_CONSTEXPR
#undef XeHE_USE_MAYBE_UNUSED
#undef XeHE_USE_NODISCARD
#endif

#endif // #if XeHE_COMPILER == XeHE_COMPILER_MSVC


#endif  //#ifndef XeHE_MSVC_H
