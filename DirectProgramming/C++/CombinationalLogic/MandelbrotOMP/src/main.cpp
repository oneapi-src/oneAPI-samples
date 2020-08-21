//==============================================================
//
// Copyright 2020 Intel Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// ===============================================================

// Initial conditions: rectangle (for image) = { (-2.5, -0.875), (1, 0.875) }
//                     height = 1024
//                     width = 2048
//                     max_depth = 100
//
// Finds the mandelbrot set given initial conditions, and saves results to a png
// image. The real portion of the complex number is the x-axis, and the
// imaginary portion is the y-axis
//
// You can optionally compile with GCC and MSC, but just the linear, scalar
// version will compile and it will not have all optimizations

#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <complex>

#include "mandelbrot.hpp"
#include "timer.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

void write_image(const char* filename, int width, int height,
                 unsigned char* output) {
  stbi_write_png(filename, width, height, 1, output, width);
}

int main(int argc, char* argv[]) {
  double x0 = -2.5;
  double y0 = -0.875;
  double x1 = 1;
  double y1 = 0.875;

  // Modifiable parameters:
  int height = 1024;
  int width = 2048;  // Width should be a multiple of 8
  int max_depth = 100;

  assert(width % 8 == 0);

#ifndef __INTEL_COMPILER
  CUtilTimer timer;
  printf(
      "This example will check how many iterations of z_n+1 = z_n^2 + c a "
      "complex set will remain bounded.\n");
#ifdef PERF_NUM
  double avg_time = 0;
  for (int i = 0; i < 5; ++i) {
#endif
    printf("Starting serial, scalar Mandelbrot...\n");
    timer.start();
    unsigned char* output =
        serial_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
    timer.stop();
    printf("Calculation finished. Processing time was %.0fms\n",
           timer.get_time() * 1000.0);
    printf("Saving image...\n\n");
    write_image("mandelbrot_serial.png", width, height, output);
    _mm_free(output);
#ifdef PERF_NUM
    avg_time += timer.get_time();
  }
  printf("avg time: %.0fms\n", avg_time * 1000.0 / 5);
#endif
#else
  int option = 0;
#ifndef PERF_NUM
  // Checks to see if option was given at command line
  if (argc > 1) {
    // Prints out instructions and quits
    if (argv[1][0] == 'h') {
      printf(
          "This example will check how many iterations of z_n+1 = z_n^2 + c a "
          "complex set will remain bounded. Pick which parallel method you "
          "would like to use.\n");
      printf(
          "[0] all tests\n[1] serial/scalar\n[2] OpenMP SIMD\n[3] OpenMP "
          "Parallel\n[4] OpenMP Both\n  > ");
      return 0;
    } else {
      option = atoi(argv[1]);
    }
  }
  // If no options are given, prompt user to choose an option
  else {
    printf(
        "This example will check how many iterations of z_n+1 = z_n^2 + c a "
        "complex set will remain bounded. Pick which parallel method you would "
        "like to use.\n");
    printf(
        "[0] all tests\n[1] serial/scalar\n[2] OpenMP SIMD\n[3] OpenMP "
        "Parallel\n[4] OpenMP Both\n  > ");
    scanf("%i", &option);
  }
#endif  // !PERF_NUM

  CUtilTimer timer;
  double serial_time, omp_simd_time, omp_parallel_time, omp_both_time;
  unsigned char* output;
  switch (option) {
    case 0: {
#ifdef PERF_NUM
      double avg_time[4] = {0.0};
      for (int i = 0; i < 5; ++i) {
#endif
        printf("\nRunning all tests\n");

        printf("\nStarting serial, scalar Mandelbrot...\n");
        timer.start();
        output = serial_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
        timer.stop();
        serial_time = timer.get_time();
        printf("Calculation finished. Processing time was %.0fms\n",
               serial_time * 1000.0);
        printf("Saving image as mandelbrot_serial.png\n");
        write_image("mandelbrot_serial.png", width, height, output);
        _mm_free(output);

        printf("\nStarting OMP SIMD Mandelbrot...\n");
        timer.start();
        output = simd_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
        timer.stop();
        omp_simd_time = timer.get_time();
        printf("Calculation finished. Processing time was %.0fms\n",
               omp_simd_time * 1000.0);
        printf("Saving image as mandelbrot_simd.png\n");
        write_image("mandelbrot_simd.png", width, height, output);
        _mm_free(output);

        printf("\nStarting OMP Parallel Mandelbrot...\n");
        timer.start();
        output = parallel_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
        timer.stop();
        omp_parallel_time = timer.get_time();
        printf("Calculation finished. Processing time was %.0fms\n",
               omp_parallel_time * 1000.0);
        printf("Saving image as mandelbrot_parallel.png\n");
        write_image("mandelbrot_parallel.png", width, height, output);
        _mm_free(output);

        printf("\nStarting OMP SIMD + Parallel Mandelbrot...\n");
        timer.start();
        output = omp_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
        timer.stop();
        omp_both_time = timer.get_time();
        printf("Calculation finished. Processing time was %.0fms\n",
               omp_both_time * 1000.0);
        printf("Saving image as mandelbrot_simd_parallel.png\n");
        write_image("mandelbrot_simd_parallel.png", width, height, output);
        _mm_free(output);
#ifndef PERF_NUM
      }
#endif
#ifdef PERF_NUM
      avg_time[0] += serial_time;
      avg_time[1] += omp_simd_time;
      avg_time[2] += omp_parallel_time;
      avg_time[3] += omp_both_time;
    }
      printf("\navg time (serial)            : %.0fms\n",
             avg_time[0] * 1000.0 / 5);
      printf("avg time (simd)              : %.0fms\n",
             avg_time[1] * 1000.0 / 5);
      printf("avg time (parallel)          : %.0fms\n",
             avg_time[2] * 1000.0 / 5);
      printf("avg time (simd+parallel)     : %.0fms\n\n",
             avg_time[3] * 1000.0 / 5);
  }
#endif
  break;

  case 1: {
    printf("\nStarting serial, scalar Mandelbrot...\n");
    timer.start();
    output = serial_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
    timer.stop();
    printf("Calculation finished. Processing time was %.0fms\n",
           timer.get_time() * 1000.0);
    printf("Saving image as mandelbrot_serial.png\n");
    write_image("mandelbrot_serial.png", width, height, output);
    _mm_free(output);
    break;
  }

  case 2: {
    printf("\nStarting OMP SIMD Mandelbrot...\n");
    timer.start();
    output = simd_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
    timer.stop();
    printf("Calculation finished. Processing time was %.0fms\n",
           timer.get_time() * 1000.0);
    printf("Saving image as mandelbrot_simd.png\n");
    write_image("mandelbrot_simd.png", width, height, output);
    _mm_free(output);
    break;
  }

  case 3: {
    printf("\nStarting OMP Parallel Mandelbrot...\n");
    timer.start();
    output = parallel_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
    timer.stop();
    printf("Calculation finished. Processing time was %.0fms\n",
           timer.get_time() * 1000.0);
    printf("Saving image as mandelbrot_parallel.png\n");
    write_image("mandelbrot_parallel.png", width, height, output);
    _mm_free(output);
    break;
  }

  case 4: {
    printf("\nStarting OMP Mandelbrot...\n");
    timer.start();
    output = omp_mandelbrot(x0, y0, x1, y1, width, height, max_depth);
    timer.stop();
    printf("Calculation finished. Processing time was %.0fms\n",
           timer.get_time() * 1000.0);
    printf("Saving image as mandelbrot_simd_parallel.png\n");
    write_image("mandelbrot_simd_parallel.png", width, height, output);
    _mm_free(output);
    break;
  }

  default: {
    printf("Please pick a valid option\n");
    break;
  }
}
#endif
  return 0;
}
