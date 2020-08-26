//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/* [DESCRIPTION]
 * This C code sample demonstrates how to use C, Intel(R) MMX(TM),
 * Intel(R) Streaming SIMD Extensions 3 (Intel(R) SSE3),
 * Intel(R) Advanced Vector Extensions (Intel(R) AVX), and
 * Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)
 * intrinsics to calculate the dot product of two vectors.
 *
 * Do not run the sample on systems using processors that do
 * not support Intel(R) MMX(TM), Intel(R) SSE3; the application
 * will fail.
 *
 * [Output]
 * Dot Product computed by C:  4324.000000
 * Dot Product computed by Intel(R) SSE3 intrinsics:  4324.000000
 * Dot Product computed by Intel(R) AVX intrinsics:  4324.000000
 * Dot Product computed by Intel(R) AVX2 intrinsics:  4324.000000
 * Dot Product computed by Intel(R) MMX(TM) intrinsics:  4324
 *
 */
#include <immintrin.h>
#include <omp.h>
#include <pmmintrin.h>
#include <stdio.h>

#define SIZE 24  // assumes size is a multiple of 8 because
// Intel(R) AVX registers will store 8, 32bit elements.

// Computes dot product using C
float dot_product(float *a, float *b);
// Computes dot product using SIMD
float dot_product_SIMD(float *a, float *b);
// Computes dot product using Intel(R) SSE intrinsics
float dot_product_intrin(float *a, float *b);
// Computes dot product using Intel(R) AVX intrinsics
float AVX_dot_product(float *a, float *b);
float AVX2_dot_product(float *a, float *b);
// Computes dot product using Intel(R) MMX(TM) intrinsics
short MMX_dot_product(short *a, short *b);

#define MMX_DOT_PROD_ENABLED (__INTEL_COMPILER || (_MSC_VER && !_WIN64))

int main() {
  float x[SIZE], y[SIZE];
  short a[SIZE], b[SIZE];
  int i;
  float product;
  short mmx_product;
  for (i = 0; i < SIZE; i++) {
    x[i] = i;
    y[i] = i;
    a[i] = i;
    b[i] = i;
  }

  product = dot_product(x, y);
  printf("Dot Product computed by C:  %f\n", product);

  product = dot_product_SIMD(x, y);
  printf("Dot Product computed by C + SIMD:  %f\n", product);

  product = dot_product_intrin(x, y);
  printf("Dot Product computed by Intel(R) SSE3 intrinsics:  %f\n", product);

  // The Visual Studio* editor will show the following section as disabled as it
  // does not know that __INTEL_COMPILER is defined by the Intel (R) Compiler
#if __INTEL_COMPILER
  if (_may_i_use_cpu_feature(_FEATURE_AVX2)) {
    product = AVX2_dot_product(x, y);
    printf("Dot Product computed by Intel(R) AVX2 intrinsics:  %f\n", product);
  } else
    printf("Your Processor does not support AVX2 instrinsics.\n");
  if (_may_i_use_cpu_feature(_FEATURE_AVX)) {
    product = AVX_dot_product(x, y);
    printf("Dot Product computed by Intel(R) AVX intrinsics:  %f\n", product);
  } else
    printf("Your Processor does not support AVX intrinsics.\n");
#else
  printf("Use Intel(R) Compiler to compute with Intel(R) AVX intrinsics\n");
#endif

#if MMX_DOT_PROD_ENABLED
  mmx_product = MMX_dot_product(a, b);
  _mm_empty();
  printf("Dot Product computed by Intel(R) MMX(TM) intrinsics:  %d\n",
         mmx_product);

#else
  printf(
      "Use Intel(R) compiler in order to calculate dot product using Intel(R) "
      "MMX(TM) intrinsics\n");
#endif

  return 0;
}

float dot_product(float *a, float *b) {
  int i;
  int sum = 0;
  for (i = 0; i < SIZE; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

float dot_product_SIMD(float *a, float *b) {
  int i;
  int sum = 0;
#pragma omp simd reduction(+ : sum)
  for (i = 0; i < SIZE; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

// The Visual Studio* editor will show the following section as disabled as it
// does not know that __INTEL_COMPILER is defined by the Intel(R) Compiler
#if __INTEL_COMPILER

float AVX2_dot_product(float *a, float *b) {
  float total;
  int i;
  __m256 num1, num2, num3;
  __m128 top, bot;
  num3 = _mm256_setzero_ps();  // sets sum to zero
  for (i = 0; i < SIZE; i += 8) {
    num1 = _mm256_loadu_ps(a + i);  // loads unaligned array a into num1
    // num1= a[7] a[6] a[5] a[4] a[3]  a[2]  a[1]  a[0]
    num2 = _mm256_loadu_ps(b + i);  // loads unaligned array b into num2
    // num2= b[7] b[6] b[5] b[4] b[3]   b[2]   b[1]  b[0]
    num3 = _mm256_fmadd_ps(
        num1, num2, num3);  // performs multiplication and vertical addition
    // num3 = a[7]*b[7]+num3[7]  a[6]*b[6]+num3[6]  a[5]*b[5]+num3[5]
    // a[4]*b[4]+num3[4]
    //       a[3]*b[3]+num3[3]  a[2]*b[2]+num3[2]  a[1]*b[1]+num3[1]
    //       a[0]*b[0]+num3[0]
  }
  num3 = _mm256_hadd_ps(num3, num3);  // performs horizontal addition
  // For example, if num3 is filled with: 7 6 5 4 3 2 1 0
  // then num3 = 13 9 13 9 5 1 5 1

  // extracting the __m128 from the __m256 datatype
  top = _mm256_extractf128_ps(num3, 1);  // top = 13 9 13 9
  bot = _mm256_extractf128_ps(num3, 0);  // bot = 5 1 5 1

  // completing the reduction
  top = _mm_add_ps(top, bot);   // top = 14 10 14 10
  top = _mm_hadd_ps(top, top);  // top = 24 24 24 24

  _mm_store_ss(&total, top);  // Storing the result in total

  return total;
}

float AVX_dot_product(float *a, float *b) {
  float total;
  int i;
  __m256 num1, num2, num3, num4;
  __m128 top, bot;
  num4 = _mm256_setzero_ps();  // sets sum to zero
  for (i = 0; i < SIZE; i += 8) {
    num1 = _mm256_loadu_ps(a + i);  // loads unaligned array a into num1
    // num1= a[7] a[6] a[5] a[4] a[3]  a[2]  a[1]  a[0]
    num2 = _mm256_loadu_ps(b + i);  // loads unaligned array b into num2
    // num2= b[7] b[6] b[5] b[4] b[3]   b[2]   b[1]  b[0]
    num3 = _mm256_mul_ps(num1, num2);  // performs multiplication
    // num3 = a[7]*b[7]  a[6]*b[6]  a[5]*b[5]  a[4]*b[4]  a[3]*b[3]  a[2]*b[2]
    // a[1]*b[1]  a[0]*b[0]
    num4 = _mm256_add_ps(num4, num3);  // performs vertical addition
  }
  num4 = _mm256_hadd_ps(num4, num4);  // performs horizontal addition
  // For example, if num4 is filled with: 7 6 5 4 3 2 1 0
  // then num4 = 13 9 13 9 5 1 5 1

  // extracting the __m128 from the __m256 datatype
  top = _mm256_extractf128_ps(num4, 1);  // top = 13 9 13 9
  bot = _mm256_extractf128_ps(num4, 0);  // bot = 5 1 5 1

  // completing the reduction
  top = _mm_add_ps(top, bot);   // top = 14 10 14 10
  top = _mm_hadd_ps(top, top);  // top = 24 24 24 24

  _mm_store_ss(&total, top);  // Storing the result in total

  return total;
}
#endif

float dot_product_intrin(float *a, float *b) {
  float total;
  int i;
  __m128 num1, num2, num3, num4;
  __m128 num5;
  num4 = _mm_setzero_ps();  // sets sum to zero
  for (i = 0; i < SIZE; i += 4) {
    num1 = _mm_loadu_ps(
        a +
        i);  // loads unaligned array a into num1  num1= a[3]  a[2]  a[1]  a[0]
    num2 = _mm_loadu_ps(
        b +
        i);  // loads unaligned array b into num2  num2= b[3]   b[2]   b[1] b[0]
    num3 = _mm_mul_ps(num1, num2);  // performs multiplication   num3 =
                                    // a[3]*b[3]  a[2]*b[2]  a[1]*b[1] a[0]*b[0]
    num3 = _mm_hadd_ps(num3, num3);  // performs horizontal addition
    // num3=  a[3]*b[3]+ a[2]*b[2]  a[1]*b[1]+a[0]*b[0]  a[3]*b[3]+ a[2]*b[2]
    // a[1]*b[1]+a[0]*b[0]
    num4 = _mm_add_ps(num4, num3);  // performs vertical addition
  }

  num4 = _mm_hadd_ps(num4, num4);
  _mm_store_ss(&total, num4);
  return total;
}

// Intel(R) MMX(TM) technology cannot handle single precision floats
#if MMX_DOT_PROD_ENABLED
short MMX_dot_product(short *a, short *b) {
  int i;
  short result, data;
  __m64 num3, sum;
  __m64 *ptr1, *ptr2;
  _m_empty();
  sum = _mm_setzero_si64();  // sets sum to zero
  for (i = 0; i < SIZE; i += 4) {
    ptr1 = (__m64 *)&a[i];  // Converts array a to a pointer of type
    //__m64 and stores four elements into
    // Intel(R) MMX(TM) registers
    ptr2 = (__m64 *)&b[i];
    num3 = _m_pmaddwd(*ptr1, *ptr2);  // multiplies elements and adds lower
    // elements with lower element and
    // higher elements with higher
    sum = _m_paddw(sum, num3);
  }

  data = _m_to_int(sum);     // converts __m64 data type to an int
  sum = _m_psrlqi(sum, 32);  // shifts sum
  result = _m_to_int(sum);
  result = result + data;
  _mm_empty();  // clears the Intel(R) MMX(TM) registers and
  // Intel(R) MMX(TM) state.
  return result;
}
#endif
