//==============================================================
// Copyright   2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include <chrono>
#include <string>
#include <fstream>

#include "Utils.hpp"

using namespace sycl;

void iso3dfd(queue& q, float* next, float* prev, float* vel, float* coeff,
             const size_t n1, const size_t n2, const size_t n3,
             const size_t nreps) {
  // Create 2D SYCL range for kernels which not include HALO
  range<2> kernel_range((n1 - 2 * kHalfLength),
                        (n2 - 2 * kHalfLength));
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
      // Each kernel runs single row over third dimension
      h.parallel_for(kernel_range, [=](id<2> nidx) {
            // Start of device code
            // Add offsets to indices to exclude HALO
            // Start and end index used in loop
            int n2n3 = n2 * n3;
            int i = nidx[0] + kHalfLength;
            int j = nidx[1] + kHalfLength;
            int k = kHalfLength;
            int end_k = n3 - kHalfLength;

            // Calculate global linear index for each cell
            int idx = i * n2n3 + j * n3 + k;

            // Create arrays to store data used multiple times
            // Local copy of coeff buffer and continous values over third dimension which are
            // used to calculate stencil front and back arrays are used to
            // ensure the values over third dimension are read once, shifted in`
            // these array and re-used multiple times before being discarded
            // This is an optimization technique to enable data-reuse and
            // improve overall FLOPS to BYTES read ratio
            float coeff[kHalfLength + 1];
            float front[kHalfLength + 1];
            float back[kHalfLength];

            // Fill local arrays, front[0] contains current cell value
            for (int x = 0; x <= kHalfLength; x++) {
              coeff[x] = coeff_acc[x];
              front[x] = prev_acc[idx + x];
            }
            for (int x = 1; x <= kHalfLength; x++) {
              back[x-1] = prev_acc[idx - x];
            }

            // Iterate over third dimension excluding HALO
            for (; k < end_k; k++) {
        
              // Calculate values for each cell
              float value = front[0] * coeff[0];
              #pragma unroll(kHalfLength)
              for (int x = 1; x <= kHalfLength; x++) {
                value += coeff[x] *
                             (front[x] + back[x - 1] +
                              prev_acc[idx + x * n3]   + prev_acc[idx - x * n3] +
                              prev_acc[idx + x * n2n3] + prev_acc[idx - x * n2n3]);
              }
              next_acc[idx] = 2.0f * front[0] - next_acc[idx] +
                                  value * vel_acc[idx];

              // Increase linear index, jump to the next cell in third dimension
              idx++;

              // Shift values in front and back arrays
              for (auto x = kHalfLength - 1; x > 0; x--) {
                back[x] = back[x - 1];
              }
              back[0] = front[0];

              for (auto x = 0; x < kHalfLength; x++) {
                front[x] = front[x + 1];
              }
              front[kHalfLength] = prev_acc[idx + kHalfLength];
            }
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

  // Variables to store size of grids, number of simulation iterations,
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
  std::cout << " Running GPU private memory version with iterations over third dimension\n";
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