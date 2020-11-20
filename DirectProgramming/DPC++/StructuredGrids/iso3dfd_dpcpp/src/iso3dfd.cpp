//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// ISO3DFD: Intel® oneAPI DPC++ Language Basics Using 3D-Finite-Difference-Wave
// Propagation
//
// ISO3DFD is a finite difference stencil kernel for solving the 3D acoustic
// isotropic wave equation. Kernels in this sample are implemented as 16th order
// in space, 2nd order in time scheme without boundary conditions. Using Data
// Parallel C++, the sample can explicitly run on the GPU and/or CPU to
// calculate a result.  If successful, the output will print the device name
// where the DPC++ code ran along with the grid computation metrics - flops
// and effective throughput
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide
// and search based on relevant terms noted in the comments.
//
// DPC++ material used in this code sample:
//
// DPC++ Queues (including device selectors and exception handlers)
// DPC++ Custom device selector
// DPC++ Buffers and accessors (communicate data between the host and the
// device)
// DPC++ Kernels (including parallel_for function and nd-range<3>
// objects)
// Shared Local Memory (SLM) optimizations (DPC++)
// DPC++ Basic synchronization (barrier function)
//
#include "iso3dfd.h"
#include <iostream>
#include "device_selector.hpp"
#include "dpc_common.hpp"

/*
 * Host-Code
 * Function used for initialization
 */
void Initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, size_t n1,
                size_t n2, size_t n3) {
  std::cout << "Initializing ... \n";
  size_t dim2 = n2 * n1;

  for (size_t i = 0; i < n3; i++) {
    for (size_t j = 0; j < n2; j++) {
      size_t offset = i * dim2 + j * n1;
#pragma omp simd
      for (int k = 0; k < n1; k++) {
        ptr_prev[offset + k] = 0.0f;
        ptr_next[offset + k] = 0.0f;
        ptr_vel[offset + k] =
            2250000.0f * dt * dt;  // Integration of the v*v and dt*dt
      }
    }
  }
  // Add a source to initial wavefield as an initial condition
  float val = 1.f;
  for (int s = 5; s >= 0; s--) {
    for (int i = n3 / 2 - s; i < n3 / 2 + s; i++) {
      for (int j = n2 / 4 - s; j < n2 / 4 + s; j++) {
        size_t offset = i * dim2 + j * n1;
        for (int k = n1 / 4 - s; k < n1 / 4 + s; k++) {
          ptr_prev[offset + k] = val;
        }
      }
    }
    val *= 10;
  }
}

/*
 * Host-Code
 * OpenMP implementation for single iteration of iso3dfd kernel.
 * This function is used as reference implementation for verification and
 * also to compare performance of OpenMP and DPC++ on CPU
 * Additional Details:
 * https://software.intel.com/en-us/articles/eight-optimizations-for-3-dimensional-finite-difference-3dfd-code-with-an-isotropic-iso
 */
void Iso3dfdIteration(float* ptr_next_base, float* ptr_prev_base,
                      float* ptr_vel_base, float* coeff, const size_t n1,
                      const size_t n2, const size_t n3, const size_t n1_block,
                      const size_t n2_block, const size_t n3_block) {
  size_t dimn1n2 = n1 * n2;
  size_t n3End = n3 - kHalfLength;
  size_t n2End = n2 - kHalfLength;
  size_t n1End = n1 - kHalfLength;

#pragma omp parallel default(shared)
#pragma omp for schedule(static) collapse(3)
  for (size_t bz = kHalfLength; bz < n3End;
       bz += n3_block) {  // start of cache blocking
    for (size_t by = kHalfLength; by < n2End; by += n2_block) {
      for (size_t bx = kHalfLength; bx < n1End; bx += n1_block) {
        int izEnd = std::min(bz + n3_block, n3End);
        int iyEnd = std::min(by + n2_block, n2End);
        int ixEnd = std::min(n1_block, n1End - bx);
        for (size_t iz = bz; iz < izEnd; iz++) {  // start of inner iterations
          for (size_t iy = by; iy < iyEnd; iy++) {
            float* ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1 + bx;
            float* ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1 + bx;
            float* ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1 + bx;
#pragma omp simd
            for (size_t ix = 0; ix < ixEnd; ix++) {
              float value = 0.0;
              value += ptr_prev[ix] * coeff[0];
#pragma unroll(kHalfLength)
              for (unsigned int ir = 1; ir <= kHalfLength; ir++) {
                value += coeff[ir] *
                         ((ptr_prev[ix + ir] + ptr_prev[ix - ir]) +
                          (ptr_prev[ix + ir * n1] + ptr_prev[ix - ir * n1]) +
                          (ptr_prev[ix + ir * dimn1n2] +
                           ptr_prev[ix - ir * dimn1n2]));
              }
              ptr_next[ix] =
                  2.0f * ptr_prev[ix] - ptr_next[ix] + value * ptr_vel[ix];
            }
          }
        }  // end of inner iterations
      }
    }
  }  // end of cache blocking
}

/*
 * Host-Code
 * Driver function for ISO3DFD OpenMP code
 * Uses ptr_next and ptr_prev as ping-pong buffers to achieve
 * accelerated wave propogation
 */
void Iso3dfd(float* ptr_next, float* ptr_prev, float* ptr_vel, float* coeff,
             const size_t n1, const size_t n2, const size_t n3,
             const unsigned int nreps, const size_t n1_block,
             const size_t n2_block, const size_t n3_block) {
  for (unsigned int it = 0; it < nreps; it += 1) {
    Iso3dfdIteration(ptr_next, ptr_prev, ptr_vel, coeff, n1, n2, n3, n1_block,
                     n2_block, n3_block);

    // here's where boundary conditions and halo exchanges happen
    // Swap previous & next between iterations
    it++;
    if (it < nreps)
      Iso3dfdIteration(ptr_prev, ptr_next, ptr_vel, coeff, n1, n2, n3, n1_block,
                       n2_block, n3_block);
  }  // time loop
}

/*
 * Host-Code
 * Main function to drive the sample application
 */
int main(int argc, char* argv[]) {
  // Arrays used to update the wavefield
  float* prev_base;
  float* next_base;
  // Array to store wave velocity
  float* vel_base;
  // Array to store results for comparison
  float* temp;

  bool sycl = true;
  bool omp = true;
  bool error = false;
  bool is_gpu = true;

  size_t n1, n2, n3;
  size_t n1_block, n2_block, n3_block;
  unsigned int num_iterations;

  // Read Input Parameters
  try {
    n1 = std::stoi(argv[1]) + (2 * kHalfLength);
    n2 = std::stoi(argv[2]) + (2 * kHalfLength);
    n3 = std::stoi(argv[3]) + (2 * kHalfLength);
    n1_block = std::stoi(argv[4]);
    n2_block = std::stoi(argv[5]);
    n3_block = std::stoi(argv[6]);
    num_iterations = std::stoi(argv[7]);
  }

  catch (...) {
    Usage(argv[0]);
    return 1;
  }

  // Read optional arguments to select version and device
  for (auto arg = 8; arg < argc; arg++) {
    std::string arg_value = argv[arg];
    std::transform(arg_value.begin(), arg_value.end(), arg_value.begin(), ::tolower);

    if (arg_value == "omp") {
      omp = true;
      sycl = false;
    } else if (arg_value == "sycl") {
      omp = false;
      sycl = true;
    } else if (arg_value == "gpu") {
      is_gpu = true;
    } else if (arg_value == "cpu") {
      is_gpu = false;
    } else {
      Usage(argv[0]);
      return 1;
    }
  }

  // Validate input sizes for the grid and block dimensions
  if (CheckGridDimension(n1 - 2 * kHalfLength, n2 - 2 * kHalfLength,
                         n3 - 2 * kHalfLength, n1_block, n2_block, n3_block)) {
    Usage(argv[0]);
    return 1;
  }

  // Compute the total size of grid
  size_t nsize = n1 * n2 * n3;

  prev_base = new float[nsize];
  next_base = new float[nsize];
  vel_base = new float[nsize];

  // Compute coefficients to be used in wavefield update
  float coeff[kHalfLength + 1] = {-3.0548446,   +1.7777778,     -3.1111111e-1,
                                  +7.572087e-2, -1.76767677e-2, +3.480962e-3,
                                  -5.180005e-4, +5.074287e-5,   -2.42812e-6};

  // Apply the DX DY and DZ to coefficients
  coeff[0] = (3.0f * coeff[0]) / (dxyz * dxyz);
  for (int i = 1; i <= kHalfLength; i++) {
    coeff[i] = coeff[i] / (dxyz * dxyz);
  }

  std::cout << "Grid Sizes: " << n1 - 2 * kHalfLength << " "
            << n2 - 2 * kHalfLength << " " << n3 - 2 * kHalfLength << "\n";
  std::cout << "Memory Usage: " << ((3 * nsize * sizeof(float)) / (1024 * 1024))
            << " MB\n";

  // Check if running OpenMP OR Serial version on CPU
  if (omp) {
#if defined(_OPENMP)
    std::cout << " ***** Running OpenMP variant *****\n";
#else
    std::cout << " ***** Running C++ Serial variant *****\n";
#endif

    // Initialize arrays and introduce initial conditions (source)
    Initialize(prev_base, next_base, vel_base, n1, n2, n3);

    // Start timer
    dpc_common::TimeInterval t_ser;
    // Invoke the driver function to perform 3D wave propogation
    // using OpenMP/Serial version
    Iso3dfd(next_base, prev_base, vel_base, coeff, n1, n2, n3, num_iterations,
            n1_block, n2_block, n3_block);

    // End timer
    PrintStats(t_ser.Elapsed() * 1e3, n1, n2, n3, num_iterations);
  }

  // Check if running both OpenMP/Serial and DPC++ version
  // Keeping a copy of output buffer from OpenMP version
  // for comparison
  if (omp && sycl) {
    temp = new float[nsize];
    if (num_iterations % 2)
      memcpy(temp, next_base, nsize * sizeof(float));
    else
      memcpy(temp, prev_base, nsize * sizeof(float));
  }

  // Check if running DPC++/SYCL version
  if (sycl) {
    std::cout << " ***** Running SYCL variant *****\n";
    // Initialize arrays and introduce initial conditions (source)
    Initialize(prev_base, next_base, vel_base, n1, n2, n3);

    // Initializing a string pattern to allow a custom device selector
    // pick a SYCL device as per user's preference and available devices
    // Default value of pattern is set to CPU
    std::string pattern("CPU");
    std::string pattern_gpu("Gen");

    // Replacing the pattern string to Gen if running on a GPU
    if (is_gpu) {
      pattern.replace(0, 3, pattern_gpu);
    }

    // Create a custom device selector using DPC++ device selector class
    MyDeviceSelector device_sel(pattern);

    // Create a device queue using DPC++ class queue with a custom
    // device selector
    queue q(device_sel, dpc_common::exception_handler);

    // Validate if the block sizes selected are
    // within range for the selected SYCL device
    if (CheckBlockDimension(q, n1_block, n2_block)) {
      Usage(argv[0]);
      return 1;
    }

    // Start timer
    dpc_common::TimeInterval t_dpc;

    // Invoke the driver function to perform 3D wave propogation
    // using DPC++ version on the selected SYCL device
    Iso3dfdDevice(q, next_base, prev_base, vel_base, coeff, n1, n2, n3,
                  n1_block, n2_block, n3_block, n3 - kHalfLength,
                  num_iterations);
    // Wait for the commands to complete. Enforce synchronization on the command
    // queue
    q.wait_and_throw();

    // End timer
    PrintStats(t_dpc.Elapsed() * 1e3, n1, n2, n3, num_iterations);
  }

  // If running both OpenMP/Serial and DPC++ version
  // Comparing results
  if (omp && sycl) {
    if (num_iterations % 2) {
      error = WithinEpsilon(next_base, temp, n1, n2, n3, kHalfLength, 0, 0.1f);
    } else {
      error = WithinEpsilon(prev_base, temp, n1, n2, n3, kHalfLength, 0, 0.1f);
    }
    if (error) {
      std::cout << "Final wavefields from SYCL device and CPU are not "
                << "equivalent: Fail\n";
    } else {
      std::cout << "Final wavefields from SYCL device and CPU are equivalent:"
                << " Success\n";
    }
    std::cout << "--------------------------------------\n";
    delete[] temp;
  }

  delete[] prev_base;
  delete[] next_base;
  delete[] vel_base;

  return error ? 1 : 0;
}
