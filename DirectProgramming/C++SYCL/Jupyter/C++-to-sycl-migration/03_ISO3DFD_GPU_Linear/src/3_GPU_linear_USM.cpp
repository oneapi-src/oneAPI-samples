//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <chrono>
#include <string>
#include <fstream>

#include "Utils.hpp"

using namespace sycl;

bool iso3dfd(sycl::queue &q, float *ptr_next, float *ptr_prev,
                   float *ptr_vel, float *ptr_coeff, size_t n1, size_t n2,
                   size_t n3, unsigned int nIterations) {
  auto nx = n1;
  auto nxy = n1*n2;
  auto grid_size = n1*n2*n3;
  auto b1 = kHalfLength;
  auto b2 = kHalfLength;
  auto b3 = kHalfLength;
  
  // Create 3D SYCL range for kernels which not include HALO
  range<3> kernel_range(n1 - 2 * kHalfLength, n2 - 2 * kHalfLength,
                        n3 - 2 * kHalfLength);

  auto next = sycl::aligned_alloc_device<float>(64, grid_size + 16, q);
  next += (16 - b1);
  q.memcpy(next, ptr_next, sizeof(float)*grid_size);
  auto prev = sycl::aligned_alloc_device<float>(64, grid_size + 16, q);
  prev += (16 - b1);
  q.memcpy(prev, ptr_prev, sizeof(float)*grid_size);
  auto vel = sycl::aligned_alloc_device<float>(64, grid_size + 16, q);
  vel += (16 - b1);
  q.memcpy(vel, ptr_vel, sizeof(float)*grid_size);
  auto coeff = sycl::aligned_alloc_device<float>(64, kHalfLength + 1, q);
  //coeff += (16 - b1);
  q.memcpy(coeff, ptr_coeff, sizeof(float)*(kHalfLength+1));
  q.wait();

  for (auto it = 0; it < nIterations; it += 1) {
    // Submit command group for execution
    q.submit([&](handler& h) {
      // Send a SYCL kernel(lambda) to the device for parallel execution
      // Each kernel runs single cell
      h.parallel_for(kernel_range, [=](id<3> idx) {
        // Start of device code
        // Add offsets to indices to exclude HALO
        int n2n3 = n2 * n3;
        int i = idx[0] + kHalfLength;
        int j = idx[1] + kHalfLength;
        int k = idx[2] + kHalfLength;

        // Calculate linear index for each cell
        int gid = i * n2n3 + j * n3 + k;
        auto value = coeff[0] * prev[gid];
          
        // Calculate values for each cell
#pragma unroll(8)
        for (int x = 1; x <= kHalfLength; x++) {
          value += coeff[x] * (prev[gid + x] + prev[gid - x] +
                               prev[gid + x * n3]   + prev[gid - x * n3] +
                               prev[gid + x * n2n3] + prev[gid - x * n2n3]);
        }
        next[gid] = 2.0f * prev[gid] - next[gid] + value * vel[gid];
          
        // End of device code
      });
    }).wait();

    // Swap the buffers for always having current values in prev buffer
    std::swap(next, prev);
  }
  q.memcpy(ptr_prev, prev, sizeof(float)*grid_size);

  sycl::free(next - (16 - b1),q);
  sycl::free(prev - (16 - b1),q);
  sycl::free(vel - (16 - b1),q);
  sycl::free(coeff,q);
  return true;
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

  // Flag to verify results with CPU version
  bool verify = false;

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
    if (argc > 5) verify = true;
  } catch (...) {
    Usage(argv[0]);
    return 1;
  }

  // Validate input sizes for the grid
  if (ValidateInput(n1, n2, n3, num_iterations)) {
    Usage(argv[0]);
    return 1;
  }

  // Create queue and print target info with default selector and in order
  // property
  queue q(default_selector_v, {property::queue::in_order()});
  std::cout << " Running linear indexed GPU version\n";
  printTargetInfo(q);

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

  auto start = std::chrono::steady_clock::now();

  // Invoke the driver function to perform 3D wave propagation offloaded to
  // the device
  iso3dfd(q, next, prev, vel, coeff, n1, n2, n3, num_iterations);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  printStats(time, n1, n2, n3, num_iterations);

  // Verify result with the CPU serial version
  if (verify) {
    VerifyResult(prev, next, vel, coeff, n1, n2, n3, num_iterations);
  }

  delete[] prev;
  delete[] next;
  delete[] vel;

  return 0;
}
