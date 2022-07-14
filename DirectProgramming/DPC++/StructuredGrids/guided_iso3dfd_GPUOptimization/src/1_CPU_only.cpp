//==============================================================
// Copyright � 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <chrono>
#include <string>

#include "Utils.hpp"

void inline iso3dfdIteration(float* ptr_next_base, float* ptr_prev_base,
                             float* ptr_vel_base, float* coeff, const size_t n1,
                             const size_t n2, const size_t n3) {
  auto dimn1n2 = n1 * n2;

  // Remove HALO from the end
  auto n3_end = n3 - kHalfLength;
  auto n2_end = n2 - kHalfLength;
  auto n1_end = n1 - kHalfLength;

  for (auto iz = kHalfLength; iz < n3_end; iz++) {
    for (auto iy = kHalfLength; iy < n2_end; iy++) {
      // Calculate start pointers for the row over X dimension
      float* ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1;
      float* ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1;
      float* ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1;

      // Iterate over X
      for (auto ix = kHalfLength; ix < n1_end; ix++) {
        // Calculate values for each cell
        float value = ptr_prev[ix] * coeff[0];
        for (int i = 1; i <= kHalfLength; i++) {
          value +=
              coeff[i] *
               (ptr_prev[ix + i] + ptr_prev[ix - i] +
                ptr_prev[ix + i * n1] + ptr_prev[ix - i * n1] +
                ptr_prev[ix + i * dimn1n2] + ptr_prev[ix - i * dimn1n2]);
        }
        ptr_next[ix] = 2.0f * ptr_prev[ix] - ptr_next[ix] + value * ptr_vel[ix];
      }
    }
  }
}

void iso3dfd(float* next, float* prev, float* vel, float* coeff,
             const size_t n1, const size_t n2, const size_t n3,
             const size_t nreps) {
  for (auto it = 0; it < nreps; it++) {
    iso3dfdIteration(next, prev, vel, coeff, n1, n2, n3);
    // Swap the pointers for always having current values in prev array
    std::swap(next, prev);
  }
}

int main(int argc, char* argv[]) {
  // Arrays used to update the wavefield
  float* prev;
  float* next;
  // Array to store wave velocity
  float* vel;

  // Variables to store size of grids and number of simulation iterations
  size_t n1, n2, n3;
  size_t num_iterations;

  if (argc < 5) {
    Usage(argv[0]);
    return 1;
  }

  try {
    // Parse command line arguments and increase them by HALO
    n1 = std::stoi(argv[1]) + (2 * kHalfLength);
    n2 = std::stoi(argv[2]) + (2 * kHalfLength);
    n3 = std::stoi(argv[3]) + (2 * kHalfLength);
    num_iterations = std::stoi(argv[4]);
  } catch (...) {
    Usage(argv[0]);
    return 1;
  }

  // Validate input sizes for the grid
  if (ValidateInput(n1, n2, n3, num_iterations)) {
    Usage(argv[0]);
    return 1;
  }

  // Compute the total size of grid
  size_t nsize = n1 * n2 * n3;

  prev = new float[nsize];
  next = new float[nsize];
  vel = new float[nsize];

  // Compute coefficients to be used in wavefield update
  float coeff[kHalfLength + 1] = {-3.0548446,   +1.7777778,     -3.1111111e-1,
                                  +7.572087e-2, -1.76767677e-2, +3.480962e-3,
                                  -5.180005e-4, +5.074287e-5,   -2.42812e-6};

  // Apply the DX, DY and DZ to coefficients
  coeff[0] = (3.0f * coeff[0]) / (dxyz * dxyz);
  for (auto i = 1; i <= kHalfLength; i++) {
    coeff[i] = coeff[i] / (dxyz * dxyz);
  }

  // Initialize arrays and introduce initial conditions (source)
  initialize(prev, next, vel, n1, n2, n3);

  std::cout << "Running on CPU serial version\n";
  auto start = std::chrono::steady_clock::now();

  // Invoke the driver function to perform 3D wave propagation 1 thread serial
  // version
  iso3dfd(next, prev, vel, coeff, n1, n2, n3, num_iterations);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

  printStats(time, n1, n2, n3, num_iterations);

  delete[] prev;
  delete[] next;
  delete[] vel;

  return 0;
}