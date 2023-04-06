/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef XeHE_CLANG_H
#define XeHE_CLANG_H

#if XeHE_COMPILER == XeHE_COMPILER_CLANG

// We require clang >= 5
#if (__clang_major__ < 5) || not defined(__cplusplus)
#error "XeHE CPU requires __clang_major__  >= 5"
#endif

// Read in config.h
#include "util/config.h"



#endif // #if XeHE_COMPILER == XeHE_COMPILER_CLANG

#endif //#ifndef XeHE_CLANG_H
