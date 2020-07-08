//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/* [DESCRIPTION]
 * This code sample demonstrates how to use the
 * _MM_GET_FLUSH_ZERO_MODE() and _MM_GET_DENORMALS_ZERO_MODE()
 * macros to read the FTZ and DAZ flags in the control register.
 *
 * [Compile]
 * Windows*: icl
 *
 * Linux* and macOS*: icc -o <output file name>
 *
 * Turning off optimization changes the state of the registers.
 * Windows*: icl /Od
 *
 * Linux* and macOS*: icc -O0
 *
 * [Output]
 * Shows the state of the FTZ and DAZ registers.
 *
 */

#include <pmmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>
#pragma warning(disable : 4003)

int main(void) {
  /* Test the control register for flush to zero mode */
  if (_MM_GET_FLUSH_ZERO_MODE())
    printf("FTZ is set.\n");
  else
    printf("FTZ is not set.\n");

  /* Test the control register for denormals mode */
  if (_MM_GET_DENORMALS_ZERO_MODE())
    printf("DAZ is set.\n");
  else
    printf("DAZ is not set.\n");
  return (0);
}
