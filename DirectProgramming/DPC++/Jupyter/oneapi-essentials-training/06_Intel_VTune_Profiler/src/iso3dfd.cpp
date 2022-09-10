//==============================================================
// Copyright 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// ISO3DFD: Intel oneAPI DPC++ Language Basics Using 3D-Finite-Difference-Wave
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
#include "../include/iso3dfd.h"
#include <iostream>
#include "../include/device_selector.hpp"

#define MIN(a, b) (a) < (b) ? (a) : (b)

// using namespace cl::sycl;

/*
 * Host-Code
 * Function used for initialization
 */
void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, size_t n1,
                size_t n2, size_t n3) {
  std::cout << "Initializing ... " << "\n";
  size_t dim2 = n2 * n1;

  for (size_t i = 0; i < n3; i++) {
    for (size_t j = 0; j < n2; j++) {
      size_t offset = i * dim2 + j * n1;
#pragma omp simd
      for (int k = 0; k < n1; k++) {
        ptr_prev[offset + k] = 0.0f;
        ptr_next[offset + k] = 0.0f;
        ptr_vel[offset + k] =
            2250000.0f * DT * DT;  // Integration of the v*v and dt*dt
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
void iso_3dfd_it(float* ptr_next_base, float* ptr_prev_base,
                 float* ptr_vel_base, float* coeff, const size_t n1,
                 const size_t n2, const size_t n3, const size_t n1_Tblock,
                 const size_t n2_Tblock, const size_t n3_Tblock) {
  size_t dimn1n2 = n1 * n2;
  size_t n3End = n3 - HALF_LENGTH;
  size_t n2End = n2 - HALF_LENGTH;
  size_t n1End = n1 - HALF_LENGTH;

#pragma omp parallel default(shared)
#pragma omp for schedule(static) collapse(3)
  for (size_t bz = HALF_LENGTH; bz < n3End;
       bz += n3_Tblock) {  // start of cache blocking
    for (size_t by = HALF_LENGTH; by < n2End; by += n2_Tblock) {
      for (size_t bx = HALF_LENGTH; bx < n1End; bx += n1_Tblock) {
        int izEnd = MIN(bz + n3_Tblock, n3End);
        int iyEnd = MIN(by + n2_Tblock, n2End);
        int ixEnd = MIN(n1_Tblock, n1End - bx);
        for (size_t iz = bz; iz < izEnd; iz++) {  // start of inner iterations
          for (size_t iy = by; iy < iyEnd; iy++) {
            float* ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1 + bx;
            float* ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1 + bx;
            float* ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1 + bx;
#pragma omp simd
            for (size_t ix = 0; ix < ixEnd; ix++) {
              float value = 0.0;
              value += ptr_prev[ix] * coeff[0];
#pragma unroll(HALF_LENGTH)
              for (unsigned int ir = 1; ir <= HALF_LENGTH; ir++) {
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
void iso_3dfd(float* ptr_next, float* ptr_prev, float* ptr_vel, float* coeff,
              const size_t n1, const size_t n2, const size_t n3,
              const unsigned int nreps, const size_t n1_Tblock,
              const size_t n2_Tblock, const size_t n3_Tblock) {
  for (unsigned int it = 0; it < nreps; it += 1) {
    iso_3dfd_it(ptr_next, ptr_prev, ptr_vel, coeff, n1, n2, n3, n1_Tblock,
                n2_Tblock, n3_Tblock);

    // here's where boundary conditions and halo exchanges happen
    // Swap previous & next between iterations
    it++;
    if (it < nreps)
      iso_3dfd_it(ptr_prev, ptr_next, ptr_vel, coeff, n1, n2, n3, n1_Tblock,
                  n2_Tblock, n3_Tblock);
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
  bool isGPU = true;

  size_t n1, n2, n3;
  size_t n1_Tblock, n2_Tblock, n3_Tblock;
  unsigned int nIterations;

  // Read Input Parameters
  try {
    n1 = std::stoi(argv[1]) + (2 * HALF_LENGTH);
    n2 = std::stoi(argv[2]) + (2 * HALF_LENGTH);
    n3 = std::stoi(argv[3]) + (2 * HALF_LENGTH);
    n1_Tblock = std::stoi(argv[4]);
    n2_Tblock = std::stoi(argv[5]);
    n3_Tblock = std::stoi(argv[6]);
    nIterations = std::stoi(argv[7]);
  }

  catch (...) {
    usage(argv[0]);
    return 1;
  }

  // Read optional arguments to select version and device
  for (unsigned int arg = 8; arg < argc; arg++) {
    if (std::string(argv[arg]) == "omp" || std::string(argv[arg]) == "OMP") {
      omp = true;
      sycl = false;
    } else if (std::string(argv[arg]) == "sycl" ||
               std::string(argv[arg]) == "SYCL") {
      omp = false;
      sycl = true;
    } else if (std::string(argv[arg]) == "gpu" ||
               std::string(argv[arg]) == "GPU") {
      isGPU = true;
    } else if (std::string(argv[arg]) == "cpu" ||
               std::string(argv[arg]) == "CPU") {
      isGPU = false;
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  // Validate input sizes for the grid and block dimensions
  if (checkGridDimension(n1 - 2 * HALF_LENGTH, n2 - 2 * HALF_LENGTH,
                         n3 - 2 * HALF_LENGTH, n1_Tblock, n2_Tblock,
                         n3_Tblock)) {
    usage(argv[0]);
    return 1;
  }

  // Compute the total size of grid
  size_t nsize = n1 * n2 * n3;

  prev_base = new float[nsize];
  next_base = new float[nsize];
  vel_base = new float[nsize];

  // Compute coefficients to be used in wavefield update
  float coeff[HALF_LENGTH + 1] = {-3.0548446,   +1.7777778,     -3.1111111e-1,
                                  +7.572087e-2, -1.76767677e-2, +3.480962e-3,
                                  -5.180005e-4, +5.074287e-5,   -2.42812e-6};

  // Apply the DX DY and DZ to coefficients
  coeff[0] = (3.0f * coeff[0]) / (DXYZ * DXYZ);
  for (int i = 1; i <= HALF_LENGTH; i++) {
    coeff[i] = coeff[i] / (DXYZ * DXYZ);
  }

  std::cout << "Grid Sizes: " << n1 - 2 * HALF_LENGTH << " "
            << n2 - 2 * HALF_LENGTH << " " << n3 - 2 * HALF_LENGTH << "\n";
  std::cout << "Memory Usage: " << ((3 * nsize * sizeof(float)) / (1024 * 1024))
            << " MB" << "\n";

  // Check if running OpenMP OR Serial version on CPU
  if (omp) {
#if defined(_OPENMP)
    std::cout << " ***** Running OpenMP variant *****" << "\n";
#else
    std::cout << " ***** Running C++ Serial variant *****" << "\n";
#endif

    // Initialize arrays and introduce initial conditions (source)
    initialize(prev_base, next_base, vel_base, n1, n2, n3);

    // Start timer
    auto start = std::chrono::steady_clock::now();

    // Invoke the driver function to perform 3D wave propogation
    // using OpenMP/Serial version
    iso_3dfd(next_base, prev_base, vel_base, coeff, n1, n2, n3, nIterations,
             n1_Tblock, n2_Tblock, n3_Tblock);

    // End timer
    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    printStats(time, n1, n2, n3, nIterations);
  }

  // Check if running both OpenMP/Serial and DPC++ version
  // Keeping a copy of output buffer from OpenMP version
  // for comparison
  if (omp && sycl) {
    temp = new float[nsize];
    if (nIterations % 2)
      memcpy(temp, next_base, nsize * sizeof(float));
    else
      memcpy(temp, prev_base, nsize * sizeof(float));
  }

  // Check if running DPC++/SYCL version
  if (sycl) {
    std::cout << " ***** Running SYCL variant *****" << "\n";
    // exception handler
    /*
      The exception_list parameter is an iterable list of std::exception_ptr
      objects. But those pointers are not always directly readable. So, we
      rethrow the pointer, catch it,  and then we have the exception itself.
      Note: depending upon the operation there may be several exceptions.
    */
    auto exception_handler = [](exception_list exceptionList) {
      for (std::exception_ptr const& e : exceptionList) {
        try {
          std::rethrow_exception(e);
        } catch (exception const& e) {
          std::terminate();
        }
      }
    };

    // Initialize arrays and introduce initial conditions (source)
    initialize(prev_base, next_base, vel_base, n1, n2, n3);

    // Initializing a string pattern to allow a custom device selector
    // pick a SYCL device as per user's preference and available devices
    // Default value of pattern is set to CPU
    std::string pattern("CPU");
    std::string patterngpu("Graphics");

    // Replacing the pattern string to Gen if running on a GPU
    if (isGPU) {
      pattern.replace(0, 3, patterngpu);
    }

    // Create a custom device selector using DPC++ device selector class
    MyDeviceSelector device_sel(pattern);

    // Create a device queue using DPC++ class queue with a custom
    // device selector
    queue q(device_sel, exception_handler);

    // Validate if the block sizes selected are
    // within range for the selected SYCL device
    if (checkBlockDimension(q, n1_Tblock, n2_Tblock)) {
      usage(argv[0]);
      return 1;
    }

    // Start timer
    auto start = std::chrono::steady_clock::now();

    // Invoke the driver function to perform 3D wave propogation
    // using DPC++ version on the selected SYCL device
    iso_3dfd_device(q, next_base, prev_base, vel_base, coeff, n1, n2, n3,
                    n1_Tblock, n2_Tblock, n3_Tblock, n3 - HALF_LENGTH,
                    nIterations);
    // Wait for the commands to complete. Enforce synchronization on the command
    // queue
    q.wait_and_throw();

    // End timer
    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "SYCL time: " << time << " ms" << "\n";

    printStats(time, n1, n2, n3, nIterations);
  }

  // If running both OpenMP/Serial and DPC++ version
  // Comparing results
  if (omp && sycl) {
    if (nIterations % 2) {
      error = within_epsilon(next_base, temp, n1, n2, n3, HALF_LENGTH, 0, 0.1f);
      if (error) std::cout << "Error  = " << error << "\n";
    } else {
      error = within_epsilon(prev_base, temp, n1, n2, n3, HALF_LENGTH, 0, 0.1f);
      if (error) std::cout << "Error  = " << error << "\n";
    }
    delete[] temp;
  }

  delete[] prev_base;
  delete[] next_base;
  delete[] vel_base;

  return error ? 1 : 0;
}

