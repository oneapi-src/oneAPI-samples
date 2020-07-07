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

// Each of these methods calculate how deeply numbers on a complex plane remains
// in the Mandelbrot set. On top of the serial/scalar version, there is a
// cilk_for version, a pragma simd version, and a combined cilk_for/pragma simd
// version

#include "mandelbrot.hpp"

#include <complex>
#ifdef __INTEL_COMPILER
#include <omp.h>
#endif
#include <emmintrin.h>
// Description:
// Determines how deeply points in the complex plane, spaced on a uniform grid,
// remain in the Mandelbrot set. The uniform grid is specified by the rectangle
// (x1, y1) - (x0, y0). Mandelbrot set is determined by remaining bounded after
// iteration of z_n+1 = z_n^2 + c, up to max_depth.
//
// Everything is done in a linear, scalar fashion
//
// [in]: x0, y0, x1, y1, width, height, max_depth
// [out]: output (caller must deallocate)
unsigned char* serial_mandelbrot(double x0, double y0, double x1, double y1,
                                 int width, int height, int max_depth) {
  double xstep = (x1 - x0) / width;
  double ystep = (y1 - y0) / height;
  unsigned char* output = static_cast<unsigned char*>(
      _mm_malloc(width * height * sizeof(unsigned char), 64));

  // Traverse the sample space in equally spaced steps with width * height
  // samples
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      double z_real = x0 + i * xstep;
      double z_imaginary = y0 + j * ystep;
      double c_real = z_real;
      double c_imaginary = z_imaginary;

      // depth should be an int, but the vectorizer will not vectorize,
      // complaining about mixed data types switching it to double is worth the
      // small cost in performance to let the vectorizer work
      double depth = 0;
      // Figures out how many recurrences are required before divergence, up to
      // max_depth
      while (depth < max_depth) {
        if (z_real * z_real + z_imaginary * z_imaginary > 4.0) {
          break;  // Escape from a circle of radius 2
        }
        double temp_real = z_real * z_real - z_imaginary * z_imaginary;
        double temp_imaginary = 2.0 * z_real * z_imaginary;
        z_real = c_real + temp_real;
        z_imaginary = c_imaginary + temp_imaginary;

        ++depth;
      }
      output[j * width + i] = static_cast<unsigned char>(
          static_cast<double>(depth) / max_depth * 255);
    }
  }
  return output;
}

#ifdef __INTEL_COMPILER

#define NUM_THREADS \
  8  // USER: Experiment with various threadcounts for parallelization

// Description:
// Determines how deeply points in the complex plane, spaced on a uniform grid,
// remain in the Mandelbrot set. The uniform grid is specified by the rectangle
// (x1, y1) - (x0, y0). Mandelbrot set is determined by remaining bounded after
// iteration of z_n+1 = z_n^2 + c, up to max_depth.
//
// Optimized with OpenMP's SIMD constructs.
//
// [in]: x0, y0, x1, y1, width, height, max_depth
// [out]: output (caller must deallocate)
unsigned char* simd_mandelbrot(double x0, double y0, double x1, double y1,
                               int width, int height, int max_depth) {
  double xstep = (x1 - x0) / width;
  double ystep = (y1 - y0) / height;
  unsigned char* output = static_cast<unsigned char*>(
      _mm_malloc(width * height * sizeof(unsigned char), 64));

  // Traverse the sample space in equally spaced steps with width * height
  // samples
  for (int j = 0; j < height; ++j) {
#pragma omp simd  // vectorize code
    for (int i = 0; i < width; ++i) {
      double z_real = x0 + i * xstep;
      double z_imaginary = y0 + j * ystep;
      double c_real = z_real;
      double c_imaginary = z_imaginary;

      // depth should be an int, but the vectorizer will not vectorize,
      // complaining about mixed data types switching it to double is worth the
      // small cost in performance to let the vectorizer work
      double depth = 0;
      // Figures out how many recurrences are required before divergence, up to
      // max_depth
      while (depth < max_depth) {
        if (z_real * z_real + z_imaginary * z_imaginary > 4.0) {
          break;  // Escape from a circle of radius 2
        }
        double temp_real = z_real * z_real - z_imaginary * z_imaginary;
        double temp_imaginary = 2.0 * z_real * z_imaginary;
        z_real = c_real + temp_real;
        z_imaginary = c_imaginary + temp_imaginary;

        ++depth;
      }
      output[j * width + i] = static_cast<unsigned char>(
          static_cast<double>(depth) / max_depth * 255);
    }
  }
  return output;
}

// Description:
// Determines how deeply points in the complex plane, spaced on a uniform grid,
// remain in the Mandelbrot set. The uniform grid is specified by the rectangle
// (x1, y1) - (x0, y0). Mandelbrot set is determined by remaining bounded after
// iteration of z_n+1 = z_n^2 + c, up to max_depth.
//
// Optimized with OpenMP's parallelization constructs.
//
// [in]: x0, y0, x1, y1, width, height, max_depth
// [out]: output (caller must deallocate)
unsigned char* parallel_mandelbrot(double x0, double y0, double x1, double y1,
                                   int width, int height, int max_depth) {
  double xstep = (x1 - x0) / width;
  double ystep = (y1 - y0) / height;
  unsigned char* output = static_cast<unsigned char*>(
      _mm_malloc(width * height * sizeof(unsigned char), 64));

  omp_set_num_threads(NUM_THREADS);
  // Traverse the sample space in equally spaced steps with width * height
  // samples
#pragma omp parallel for schedule( \
    dynamic, 1)  // USER: Experiment with static/dynamic partitioning
  // dynamic partitioning is advantageous as the while loop for calculating
  // depth makes iterations vary in terms of time.
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      double z_real = x0 + i * xstep;
      double z_imaginary = y0 + j * ystep;
      double c_real = z_real;
      double c_imaginary = z_imaginary;

      // depth should be an int, but the vectorizer will not vectorize,
      // complaining about mixed data types switching it to double is worth the
      // small cost in performance to let the vectorizer work
      double depth = 0;
      // Figures out how many recurrences are required before divergence, up to
      // max_depth
      while (depth < max_depth) {
        if (z_real * z_real + z_imaginary * z_imaginary > 4.0) {
          break;  // Escape from a circle of radius 2
        }
        double temp_real = z_real * z_real - z_imaginary * z_imaginary;
        double temp_imaginary = 2.0 * z_real * z_imaginary;
        z_real = c_real + temp_real;
        z_imaginary = c_imaginary + temp_imaginary;

        ++depth;
      }
      output[j * width + i] = static_cast<unsigned char>(
          static_cast<double>(depth) / max_depth * 255);
    }
  }
  return output;
}

// Description:
// Determines how deeply points in the complex plane, spaced on a uniform grid,
// remain in the Mandelbrot set. The uniform grid is specified by the rectangle
// (x1, y1) - (x0, y0). Mandelbrot set is determined by remaining bounded after
// iteration of z_n+1 = z_n^2 + c, up to max_depth.
//
// Optimized with OpenMP's parallelization and SIMD constructs.
//
// [in]: x0, y0, x1, y1, width, height, max_depth
// [out]: output (caller must deallocate)
unsigned char* omp_mandelbrot(double x0, double y0, double x1, double y1,
                              int width, int height, int max_depth) {
  double xstep = (x1 - x0) / width;
  double ystep = (y1 - y0) / height;
  unsigned char* output = static_cast<unsigned char*>(
      _mm_malloc(width * height * sizeof(unsigned char), 64));

  omp_set_num_threads(NUM_THREADS);
  // Traverse the sample space in equally spaced steps with width * height
  // samples
#pragma omp parallel for schedule( \
    dynamic, 1)  // USER: Experiment with static/dynamic partitioning
  // dynamic partitioning is advantageous as the while loop for calculating
  // depth makes iterations vary in terms of time.
  for (int j = 0; j < height; ++j) {
#pragma omp simd  // vectorize code
    for (int i = 0; i < width; ++i) {
      double z_real = x0 + i * xstep;
      double z_imaginary = y0 + j * ystep;
      double c_real = z_real;
      double c_imaginary = z_imaginary;

      // depth should be an int, but the vectorizer will not vectorize,
      // complaining about mixed data types switching it to double is worth the
      // small cost in performance to let the vectorizer work
      double depth = 0;
      // Figures out how many recurrences are required before divergence, up to
      // max_depth
      while (depth < max_depth) {
        if (z_real * z_real + z_imaginary * z_imaginary > 4.0) {
          break;  // Escape from a circle of radius 2
        }
        double temp_real = z_real * z_real - z_imaginary * z_imaginary;
        double temp_imaginary = 2.0 * z_real * z_imaginary;
        z_real = c_real + temp_real;
        z_imaginary = c_imaginary + temp_imaginary;

        ++depth;
      }
      output[j * width + i] = static_cast<unsigned char>(
          static_cast<double>(depth) / max_depth * 255);
    }
  }
  return output;
}

#endif  // __INTEL_COMPILER
