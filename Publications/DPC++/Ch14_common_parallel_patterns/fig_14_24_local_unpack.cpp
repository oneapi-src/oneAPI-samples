// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

using namespace sycl;

const uint32_t max_iterations = 1024;
const uint32_t Nx = 1024, Ny = 768;

struct Parameters {
  float xc;
  float yc;
  float zoom;
  float zoom_px;
  float x_span;
  float y_span;
  float x0;
  float y0;
  float dx;
  float dy;
};

void reset(
    Parameters params,
    uint32_t i,
    uint32_t j,
    uint32_t& count,
    float& cr,
    float& ci,
    float& zr,
    float& zi) {
  count = 0;
  cr = params.x0 + i * params.dx;
  ci = params.y0 + j * params.dy;
  zr = zi = 0.0f;
}

bool next_iteration(
    Parameters params,
    uint32_t i,
    uint32_t j,
    uint32_t& count,
    float& cr,
    float& ci,
    float& zr,
    float& zi,
    uint32_t* mandelbrot) {
  bool converged = false;
  if (i < Nx) {
    float next_zr = zr * zr - zi * zi;
    float next_zi = 2 * zr * zi;
    zr = next_zr + cr;
    zi = next_zi + ci;
    count++;

    // Mark that this value of i has converged
    // Output the i result for this value of i
    if (count >= max_iterations or zr * zr + zi * zi >= 4.0f) {
      converged = true;
      uint32_t px = j * Nx + i;
      mandelbrot[px] = count;
    }
  }
  return converged;
}

int main() {
  queue Q;

  // Set up parameters to control divergence, image size, etc
  Parameters params;
  params.xc = 0.0f;
  params.yc = 0.0f;
  params.zoom = 1.0f;
  params.zoom_px = pow(2.0f, 3.0f - params.zoom) * 1e-3f;
  params.x_span = Nx * params.zoom_px;
  params.y_span = Ny * params.zoom_px;
  params.x0 = params.xc - params.x_span * 0.5f;
  params.y0 = params.yc - params.y_span * 0.5f;
  params.dx = params.zoom_px;
  params.dy = params.zoom_px;

  // Initialize output on the host
  uint32_t* mandelbrot = malloc_shared<uint32_t>(Ny * Nx, Q);
  std::fill(mandelbrot, mandelbrot + Ny * Nx, 0);

  range<2> global(Ny, 8);
  range<2> local(1, 8);
  Q.parallel_for(
      nd_range<2>(global, local),
      [=](nd_item<2> it) [[intel::reqd_sub_group_size(8)]] {
        const uint32_t j = it.get_global_id(0);
        sub_group sg = it.get_sub_group();

        // Treat each row as a queue of i values to compute
        // Initially the head of the queue is at 0
        uint32_t iq = 0;

        // Initially each work-item in the sub-group works on contiguous values
        uint32_t i = iq + sg.get_local_id()[0];
        iq += sg.get_max_local_range()[0];

        // Initialize the iterator variables
        uint32_t count;
        float cr, ci, zr, zi;
        if (i < Nx) {
          reset(params, i, j, count, cr, ci, zr, zi);
        }

        // Keep iterating as long as one work-item has work to do
        while (any_of_group(sg, i < Nx)) {
          uint32_t converged =
              next_iteration(params, i, j, count, cr, ci, zr, zi, mandelbrot);
          if (any_of_group(sg, converged)) {

            // Replace pixels that have converged using an unpack
            // Pixels that haven't converged are not replaced
            uint32_t index = exclusive_scan_over_group(sg, converged, plus<>());
            i = (converged) ? iq + index : i;
            iq += reduce_over_group(sg, converged, plus<>());

            // Reset the iterator variables for the new i
            if (converged) {
              reset(params, i, j, count, cr, ci, zr, zi);
            }
          }
        }
      }).wait();

  // Produce an image as a PPM file
  constexpr uint32_t max_color = 65535;
  std::ofstream ppm;
  ppm.open("mandelbrot.ppm");
  ppm << std::string("P6").c_str() << "\n";
  ppm << Nx << "\n";
  ppm << Ny << "\n";
  ppm << max_color << "\n";
  size_t eof = ppm.tellp();
  ppm.close();
  ppm.open("mandelbrot.ppm", std::ofstream::binary | std::ofstream::app);
  ppm.seekp(eof);
  std::vector<uint16_t> colors(Nx * Ny * 3);
  for (uint32_t px = 0; px < Nx * Ny; ++px) {
    const uint16_t color = (max_iterations - mandelbrot[px]) *
        (max_color / (double)max_iterations);
    colors[3 * px + 0] = color;
    colors[3 * px + 1] = color;
    colors[3 * px + 2] = color;
  }
  ppm.write((char*)colors.data(), 2 * colors.size());
  ppm.close();

  free(mandelbrot, Q);
  return 0;
}
