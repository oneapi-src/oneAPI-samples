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
 * This C code sample demonstrates how to use C in
 * comparison with
 * Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2),
 * Intel(R) Streaming SIMD Extensions 3 (Intel(R) SSE3),
 * Intel(R) Advanced Vector Extensions (Intel(R) AVX), and
 * Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)
 * intrinsics to multiply two complex numbers.
 *
 * Do not run the sample on systems using processors that do
 * not support Intel(R) SSE2, Intel(R) SSE3; the application
 * will fail.
 *
 * [Output]
 * Complex Product(C):             23.00+ -2.00i
 * Complex Product(Intel(R) AVX2): 23.00+ -2.00i
 * Complex Product(Intel(R) AVX):  23.00+ -2.00i
 * Complex Product(Intel(R) SSE3): 23.00+ -2.00i
 * Complex Product(Intel(R) SSE2): 23.00+ -2.00i
 *
 */

#include <immintrin.h>
#include <pmmintrin.h>
#include <stdio.h>

typedef struct {
  double real;
  double img;
} complex_num;

// Multiplying complex numbers in C
void multiply_C(complex_num x, complex_num y, complex_num *z) {
  z->real = (x.real * y.real) - (x.img * y.img);
  z->img = (x.img * y.real) + (y.img * x.real);
}

// The Visual Studio* editor will show the following section as disabled as it
// does not know that __INTEL_COMPILER is defined by the Intel(R) Compiler
#if __INTEL_COMPILER

//  Multiplying complex numbers using Intel(R) AVX2 intrinsics
void multiply_AVX2(complex_num x, complex_num y, complex_num *z) {
  __m256d num1, num2, num3;
  __m128d top, bot;

  // initialize
  // num1 = [x.real,-x.img,x.img,y.img]

  num1 = _mm256_set_pd(x.real, -x.img, x.img, y.img);

  // num2 = [y.real,y.img,y.real,x.real]

  num2 = _mm256_set_pd(y.real, y.img, y.real, x.real);

  // multiply the two
  // num3 = [(x.real*y.real),(-x.img*y.img),(x.img*y.real),(y.img*x.real)]

  num3 = _mm256_mul_pd(num1, num2);

  // horizontally add
  // num3 = [(x.real*y.real-x.img*y.img),(x.real*y.real-x.img*y.img),
  //        (x.img*y.real+y.img*x.real),(x.img*y.real+y.img*x.real)]

  num3 = _mm256_hadd_pd(num3, num3);

  // permute num3 so that we have what we need to store in the lower half
  // num3 = [(x.real*y.real-x.img*y.img),(x.real*y.real-x.img*y.img),
  //        (x.img*y.real+y.img*x.real),(x.real*y.real-x.img*y.img)]

  num3 = _mm256_permute4x64_pd(num3, 0b11100110);

  // obtain the 128 bit part that we need to store
  // bot = [(x.img*y.real+y.img*x.real),(x.real*y.real-x.img*y.img)]

  bot = _mm256_extractf128_pd(num3, 0);

  // store the result in z

  _mm_storeu_pd((double *)z, bot);
}

// Multiplying complex numbers using Intel(R) AVX intrinsics
void multiply_AVX(complex_num x, complex_num y, complex_num *z) {
  __m256d num1, num2, num3;
  __m128d bot;

  // initialize
  // num1 = [x.real,-x.img,x.img,y.img]

  num1 = _mm256_set_pd(x.real, -x.img, x.img, y.img);

  // num2 = [y.real,y.img,y.real,x.real]

  num2 = _mm256_set_pd(y.real, y.img, y.real, x.real);

  // multiply the two
  // num3 = [(x.real*y.real),(-x.img*y.img),(x.img*y.real),(y.img*x.real)]

  num1 = _mm256_mul_pd(num1, num2);

  // horizontally add
  // num3 = [(x.real*y.real-x.img*y.img),(x.real*y.real-x.img*y.img),
  //        (x.img*y.real+y.img*x.real),(x.img*y.real+y.img*x.real)]

  num1 = _mm256_hadd_pd(num1, num1);

  // flip the 128 bit halves of num3 and store in num2
  // num2 = [(x.img*y.real+y.img*x.real),(x.img*y.real+y.img*x.real),
  //        (x.real*y.real-x.img*y.img),(x.real*y.real-x.img*y.img)]

  num2 = _mm256_permute2f128_pd(num1, num1, 1);

  // blend num2 and num3 together so we get what we need to store
  // num3 = [(x.real*y.real-x.img*y.img),(x.real*y.real-x.img*y.img),
  //        (x.img*y.real+y.img*x.real),(x.real*y.real-x.img*y.img)]

  num1 = _mm256_blend_pd(num1, num2, 1);

  // obtain the 128 bit part that we need to store
  // bot = [(x.img*y.real+y.img*x.real),(x.real*y.real-x.img*y.img)]

  bot = _mm256_extractf128_pd(num1, 0);

  // store the result in z

  _mm_storeu_pd((double *)z, bot);
}

#endif

// Multiplying complex numbers using Intel(R) SSE3 intrinsics
void multiply_SSE3(complex_num x, complex_num y, complex_num *z) {
  __m128d num1, num2, num3;

  // Duplicates lower vector element into upper vector element.
  //   num1: [x.real, x.real]

  num1 = _mm_loaddup_pd(&x.real);

  // Move y elements into a vector
  //   num2: [y.img, y.real]

  num2 = _mm_set_pd(y.img, y.real);

  // Multiplies vector elements
  //   num3: [(x.real*y.img), (x.real*y.real)]

  num3 = _mm_mul_pd(num2, num1);

  //   num1: [x.img, x.img]

  num1 = _mm_loaddup_pd(&x.img);

  // Swaps the vector elements
  //   num2: [y.real, y.img]

  num2 = _mm_shuffle_pd(num2, num2, 1);

  //   num2: [(x.img*y.real), (x.img*y.img)]

  num2 = _mm_mul_pd(num2, num1);

  // Adds upper vector element while subtracting lower vector element
  //   num3: [((x.real *y.img)+(x.img*y.real)),
  //          ((x.real*y.real)-(x.img*y.img))]

  num3 = _mm_addsub_pd(num3, num2);

  // Stores the elements of num3 into z

  _mm_storeu_pd((double *)z, num3);
}

// Multiplying complex numbers using Intel(R) SSE2 intrinsics

void multiply_SSE2(complex_num x, complex_num y, complex_num *z)

{
  __m128d num1, num2, num3, num4;

  // Copies a single element into the vector
  //   num1:  [x.real, x.real]

  num1 = _mm_load1_pd(&x.real);

  // Move y elements into a vector
  //   num2: [y.img, y.real]

  num2 = _mm_set_pd(y.img, y.real);

  // Multiplies vector elements
  //   num3: [(x.real*y.img), (x.real*y.real)]

  num3 = _mm_mul_pd(num2, num1);

  //   num1: [x.img, x.img]

  num1 = _mm_load1_pd(&x.img);

  // Swaps the vector elements.
  //   num2: [y.real, y.img]

  num2 = _mm_shuffle_pd(num2, num2, 1);

  //   num2: [(x.img*y.real), (x.img*y.img)]

  num2 = _mm_mul_pd(num2, num1);
  num4 = _mm_add_pd(num3, num2);
  num3 = _mm_sub_pd(num3, num2);
  num4 = _mm_shuffle_pd(num3, num4, 2);

  // Stores the elements of num4 into z

  _mm_storeu_pd((double *)z, num4);
}

int main()

{
  complex_num a, b, c;
  // Initialize complex numbers

  a.real = 3;
  a.img = 2;
  b.real = 5;
  b.img = -4;

  // Output for each: 23.00+ -2.00i

  multiply_C(a, b, &c);
  printf("Complex Product(C):             %2.2f+ %2.2fi\n", c.real, c.img);

#if __INTEL_COMPILER

  if (_may_i_use_cpu_feature(_FEATURE_AVX2)) {
    multiply_AVX2(a, b, &c);
    printf("Complex Product(Intel(R) AVX2): %2.2f+ %2.2fi\n", c.real, c.img);
  } else
    printf("Your processor does not support Intel(R) AVX2 intrinsics.\n");
  if (_may_i_use_cpu_feature(_FEATURE_AVX)) {
    multiply_AVX(a, b, &c);
    printf("Complex Product(Intel(R) AVX):  %2.2f+ %2.2fi\n", c.real, c.img);
  } else
    printf("Your processor does not support AVX intrinsics.\n");

#endif

  multiply_SSE3(a, b, &c);
  printf("Complex Product(Intel(R) SSE3): %2.2f+ %2.2fi\n", c.real, c.img);
  multiply_SSE2(a, b, &c);
  printf("Complex Product(Intel(R) SSE2): %2.2f+ %2.2fi\n", c.real, c.img);

  return 0;
}
