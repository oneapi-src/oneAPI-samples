/*
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Copyright (c) 2021, Intel Corporation
# All rights reserved.
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
*/
#ifndef _NTT_PARAMS_H_
#define _NTT_PARAMS_H_

#include <stdint.h>
#include "defines.h"

#ifndef SUB_GROUP_SIZE
#define SUB_GROUP_SIZE _SIMD_WIDTH_//inherited from defines.h
#endif

#ifndef LOG_SUB_GROUP_SIZE
#define LOG_SUB_GROUP_SIZE 3//should be log(SUB_GROUP_SIZE)
#endif

#ifndef TER_GAP_SIZE
#define TER_GAP_SIZE 8//should be greater than SUB_GROUP_SIZE
#endif

#ifndef LOG_TER_GAP_SIZE
#define LOG_TER_GAP_SIZE 3//should be log(TER_GAP_SIZE)
#endif

#ifndef LOG_LOCAL_REG_SLOTS
#define LOG_LOCAL_REG_SLOTS (LOG_TER_GAP_SIZE - LOG_SUB_GROUP_SIZE)
#endif

#ifndef LOCAL_REG_SLOTS
#define LOCAL_REG_SLOTS (1<<(LOG_LOCAL_REG_SLOTS))
#endif

#ifndef LOCAL_REG_SLOTS_HALF
#define LOCAL_REG_SLOTS_HALF ((LOCAL_REG_SLOTS)>>1)
#endif

#ifndef LOCAL_REG_NUMBERS
#define LOCAL_REG_NUMBERS (LOCAL_REG_SLOTS<<1)
#endif

//requires a x86 host platform
static inline std::size_t x86_log2(const std::size_t x) {
  std::size_t y;
#ifndef WIN32
  asm ( "bsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
#else
  y = std::size_t(std::ceil(std::ilogb(double(x))));
#endif
  return y;
}

#endif