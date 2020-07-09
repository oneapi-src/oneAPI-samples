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

#ifndef MANDELBROT_H
#define MANDELBROT_H

// Checks how many iterations of the complex quadratic polynomial z_n+1 = z_n^2
// + c keeps a set of complex numbers bounded, to a certain max depth. Mapping
// of these depths to a complex plane will result in the telltale mandelbrot set
// image Uses strictly scalar methods to calculate number of iterations (depth)
unsigned char* serial_mandelbrot(double x0, double y0, double x1, double y1,
                                 int width, int height, int max_depth);

// Checks how many iterations of the complex quadratic polynomial z_n+1 = z_n^2
// + c keeps a set of complex numbers bounded, to a certain max depth. Mapping
// of these depths to a complex plane will result in the telltale mandelbrot set
// image Uses OpenMP SIMD for optimization
unsigned char* simd_mandelbrot(double x0, double y0, double x1, double y1,
                               int width, int height, int max_depth);

// Checks how many iterations of the complex quadratic polynomial z_n+1 = z_n^2
// + c keeps a set of complex numbers bounded, to a certain max depth. Mapping
// of these depths to a complex plane will result in the telltale mandelbrot set
// image Uses OpenMP Parallelization for optimization
unsigned char* parallel_mandelbrot(double x0, double y0, double x1, double y1,
                                   int width, int height, int max_depth);

// Checks how many iterations of the complex quadratic polynomial z_n+1 = z_n^2
// + c keeps a set of complex numbers bounded, to a certain max depth Mapping of
// these depths to a complex plane will result in the telltale mandelbrot set
// image Uses OpenMP SIMD + Parallelization for optimization
unsigned char* omp_mandelbrot(double x0, double y0, double x1, double y1,
                              int width, int height, int max_depth);

#endif  // MANDELBROT_H
