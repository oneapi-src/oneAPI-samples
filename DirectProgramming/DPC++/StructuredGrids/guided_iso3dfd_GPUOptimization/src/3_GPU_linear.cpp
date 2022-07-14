//==============================================================
// Copyright � 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <chrono>
#include <string>

#include "Utils.hpp"

using namespace sycl;

void iso3dfd(queue& q, float* next, float* prev, float* vel, float* coeff,
             const size_t n1, const size_t n2, const size_t n3,
             const size_t nreps) {
  // Create 3D SYCL range for kernels which not include HALO
  range<3> kernel_range(n1 - 2 * kHalfLength, n2 - 2 * kHalfLength,
                        n3 - 2 * kHalfLength);
  // Create 1D SYCL range for buffers which include HALO
  range<1> buffer_range(n1 * n2 * n3);
  // Create buffers using SYCL class buffer
  buffer next_buf(next, buffer_range);
  buffer prev_buf(prev, buffer_range);
  buffer vel_buf(vel, buffer_range);
  buffer coeff_buf(coeff, range(kHalfLength + 1));

  for (auto it = 0; it < nreps; it++) {
    // Submit command group for execution
    q.submit([&](handler& h) {
      // Create accessors
      accessor next_acc(next_buf, h);
      accessor prev_acc(prev_buf, h);
      accessor vel_acc(vel_buf, h, read_only);
      accessor coeff_acc(coeff_buf, h, read_only);

      // Send a SYCL kernel(lambda) to the device for parallel execution
      // Each kernel runs single cell
      h.parallel_for(kernel_range, [=](id<3> nidx) {
        // Start of device code
        // Add offsets to indices to exclude HALO
        int n2n3 = n2 * n3;
        int i = nidx[0] + kHalfLength;
        int j = nidx[1] + kHalfLength;
        int k = nidx[2] + kHalfLength;

        // Calculate linear index for each cell
        int idx = i * n2n3 + j * n3 + k;

        // Calculate values for each cell
        float value = prev_acc[idx] * coeff_acc[0];
#pragma unroll(8)
        for (int x = 1; x <= kHalfLength; x++) {
          value +=
              coeff_acc[x] * (prev_acc[idx + x]        + prev_acc[idx - x] +
                              prev_acc[idx + x * n3]   + prev_acc[idx - x * n3] +
                              prev_acc[idx + x * n2n3] + prev_acc[idx - x * n2n3]);
        }
        next_acc[idx] = 2.0f * prev_acc[idx] - next_acc[idx] +
                            value * vel_acc[idx];
        // End of device code
      });
    });

    // Swap the buffers for always having current values in prev buffer
    std::swap(next_buf, prev_buf);
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
  queue q(default_selector{}, {property::queue::in_order()});
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