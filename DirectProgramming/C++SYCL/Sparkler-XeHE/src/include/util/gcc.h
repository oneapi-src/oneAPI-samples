/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef XeHE_GCC_H
#define XeHE_GCC_H

#if XeHE_COMPILER == XeHE_COMPILER_GCC

// We require GCC >= 6
#if (__GNUC__ < 6) || not defined(__cplusplus)
#pragma GCC error "XeHE CPU requires __GNUC__ >= 6"
#endif

// Read in config.h
#include "util/config.h"

#if (__GNUC__ == 6) && defined(XeHE_USE_IF_CONSTEXPR)
#pragma GCC error "g++-6 cannot compile Microsoft SEAL as C++17; set CMake build option `XeHE_USE_CXX17' to OFF"
#endif


#endif // #if XeHE_COMPILER == XeHE_COMPILER_GCC

#endif // #ifndef XeHE_GCC_H
