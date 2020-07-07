//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2010-2013 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
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
