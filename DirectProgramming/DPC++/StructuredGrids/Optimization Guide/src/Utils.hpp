//==============================================================
// Copyright � 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <CL/sycl.hpp>
#include <iostream>

#include "Iso3dfd.hpp"

void Usage(const std::string& programName, bool usedNd_ranges = false) {
  std::cout << "--------------------------------------\n";
  std::cout << " Incorrect parameters \n";
  std::cout << " Usage: ";
  std::cout << programName << " n1 n2 n3 Iterations";

  if (usedNd_ranges) std::cout << " kernel_iterations n2_WGS n3_WGS";

  std::cout << " [verify]\n\n";
  std::cout << " n1 n2 n3		: Grid sizes for the stencil\n";
  std::cout << " Iterations		: No. of timesteps.\n";

  if (usedNd_ranges) {
    std::cout
        << " kernel_iterations	: No. of cells calculated by one kernel\n";
    std::cout << " n2_WGS n3_WGS		: n2 and n3 work group sizes\n";
  }
  std::cout
      << " [verify]		: Optional: Compare results with CPU version\n";
  std::cout << "--------------------------------------\n";
  std::cout << "--------------------------------------\n";
}

bool ValidateInput(size_t n1, size_t n2, size_t n3, size_t num_iterations,
                   size_t kernel_iterations = -1, size_t n2_WGS = kHalfLength,
                   size_t n3_WGS = kHalfLength) {
  if ((n1 < kHalfLength) || (n2 < kHalfLength) || (n3 < kHalfLength) ||
      (n2_WGS < kHalfLength) || (n3_WGS < kHalfLength)) {
    std::cout << "--------------------------------------\n";
    std::cout << " Invalid grid size : n1, n2, n3, n2_WGS, n3_WGS should be "
                 "greater than "
              << kHalfLength << "\n";
    return true;
  }

  if ((n2 < n2_WGS) || (n3 < n3_WGS)) {
    std::cout << "--------------------------------------\n";
    std::cout << " Invalid work group size : n2 should be greater than n2_WGS "
                 "and n3 greater than n3_WGS\n";
    return true;
  }

  if (((n2 - 2 * kHalfLength) % n2_WGS) && kernel_iterations != -1) {
    std::cout << "--------------------------------------\n";
    std::cout << " ERROR: Invalid Grid Size: n2 should be multiple of n2_WGS - "
              << n2_WGS << "\n";
    return true;
  }
  if (((n3 - 2 * kHalfLength) % n3_WGS) && kernel_iterations != -1) {
    std::cout << "--------------------------------------\n";
    std::cout << " ERROR: Invalid Grid Size: n3 should be multiple of n3_WGS - "
              << n3_WGS << "\n";
    return true;
  }
  if (((n1 - 2 * kHalfLength) % kernel_iterations) && kernel_iterations != -1) {
    std::cout << "--------------------------------------\n";
    std::cout << " ERROR: Invalid Grid Size: n1 should be multiple of "
                 "kernel_iterations - "
              << kernel_iterations << "\n";
    return true;
  }

  return false;
}

bool CheckWorkGroupSize(sycl::queue& q, unsigned int n2_WGS,
                        unsigned int n3_WGS) {
  auto device = q.get_device();
  auto max_block_size =
      device.get_info<sycl::info::device::max_work_group_size>();

  if ((max_block_size > 1) && (n2_WGS * n3_WGS > max_block_size)) {
    std::cout << "ERROR: Invalid block sizes: n2_WGS * n3_WGS should be "
                 "less than or equal to "
              << max_block_size << "\n";
    return true;
  }

  return false;
}

void printTargetInfo(sycl::queue& q) {
  auto device = q.get_device();
  auto max_block_size =
      device.get_info<sycl::info::device::max_work_group_size>();

  auto max_exec_unit_count =
      device.get_info<sycl::info::device::max_compute_units>();

  std::cout << " Running on " << device.get_info<sycl::info::device::name>()
            << "\n";
  std::cout << " The Device Max Work Group Size is : " << max_block_size
            << "\n";
  std::cout << " The Device Max EUCount is : " << max_exec_unit_count << "\n";
}

void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, size_t n1,
                size_t n2, size_t n3) {
  auto dim2 = n2 * n1;

  for (auto i = 0; i < n3; i++) {
    for (auto j = 0; j < n2; j++) {
      auto offset = i * dim2 + j * n1;

      for (auto k = 0; k < n1; k++) {
        ptr_prev[offset + k] = 0.0f;
        ptr_next[offset + k] = 0.0f;
        ptr_vel[offset + k] =
            2250000.0f * dt * dt;  // Integration of the v*v and dt*dt here
      }
    }
  }
  // Then we add a source
  float val = 1.f;
  for (auto s = 5; s >= 0; s--) {
    for (auto i = n3 / 2 - s; i < n3 / 2 + s; i++) {
      for (auto j = n2 / 4 - s; j < n2 / 4 + s; j++) {
        auto offset = i * dim2 + j * n1;
        for (auto k = n1 / 4 - s; k < n1 / 4 + s; k++) {
          ptr_prev[offset + k] = val;
        }
      }
    }
    val *= 10;
  }
}

void printStats(double time, size_t n1, size_t n2, size_t n3,
                size_t num_iterations) {
  float throughput_mpoints = 0.0f, mflops = 0.0f, normalized_time = 0.0f;
  double mbytes = 0.0f;

  normalized_time = (double)time / num_iterations;
  throughput_mpoints = ((n1 - 2 * kHalfLength) * (n2 - 2 * kHalfLength) *
                        (n3 - 2 * kHalfLength)) /
                       (normalized_time * 1e3f);
  mflops = (7.0f * kHalfLength + 5.0f) * throughput_mpoints;
  mbytes = 12.0f * throughput_mpoints;

  std::cout << "--------------------------------------\n";
  std::cout << "time         : " << time / 1e3f << " secs\n";
  std::cout << "throughput   : " << throughput_mpoints << " Mpts/s\n";
  std::cout << "flops        : " << mflops / 1e3f << " GFlops\n";
  std::cout << "bytes        : " << mbytes / 1e3f << " GBytes/s\n";
  std::cout << "\n--------------------------------------\n";
  std::cout << "\n--------------------------------------\n";
}

bool WithinEpsilon(float* output, float* reference, const size_t dim_x,
                   const size_t dim_y, const size_t dim_z,
                   const unsigned int radius, const int zadjust = 0,
                   const float delta = 0.01f) {
  std::ofstream error_file;
  error_file.open("error_diff.txt");

  bool error = false;
  double norm2 = 0;

  for (size_t iz = 0; iz < dim_z; iz++) {
    for (size_t iy = 0; iy < dim_y; iy++) {
      for (size_t ix = 0; ix < dim_x; ix++) {
        if (ix >= radius && ix < (dim_x - radius) && iy >= radius &&
            iy < (dim_y - radius) && iz >= radius &&
            iz < (dim_z - radius + zadjust)) {
          float difference = fabsf(*reference - *output);
          norm2 += difference * difference;
          if (difference > delta) {
            error = true;
            error_file << " ERROR: " << ix << ", " << iy << ", " << iz << "   "
                       << *output << "   instead of " << *reference
                       << "  (|e|=" << difference << ")\n";
          }
        }
        ++output;
        ++reference;
      }
    }
  }

  error_file.close();
  norm2 = sqrt(norm2);
  if (error) std::cout << "error (Euclidean norm): " << norm2 << "\n";
  return error;
}

void inline iso3dfdCPUIteration(float* ptr_next_base, float* ptr_prev_base,
                                float* ptr_vel_base, float* coeff,
                                const size_t n1, const size_t n2,
                                const size_t n3) {
  auto dimn1n2 = n1 * n2;

  auto n3_end = n3 - kHalfLength;
  auto n2_end = n2 - kHalfLength;
  auto n1_end = n1 - kHalfLength;

  for (auto iz = kHalfLength; iz < n3_end; iz++) {
    for (auto iy = kHalfLength; iy < n2_end; iy++) {
      float* ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1;
      float* ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1;
      float* ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1;

      for (auto ix = kHalfLength; ix < n1_end; ix++) {
        float value = ptr_prev[ix] * coeff[0];
        value += STENCIL_LOOKUP(1);
        value += STENCIL_LOOKUP(2);
        value += STENCIL_LOOKUP(3);
        value += STENCIL_LOOKUP(4);
        value += STENCIL_LOOKUP(5);
        value += STENCIL_LOOKUP(6);
        value += STENCIL_LOOKUP(7);
        value += STENCIL_LOOKUP(8);

        ptr_next[ix] = 2.0f * ptr_prev[ix] - ptr_next[ix] + value * ptr_vel[ix];
      }
    }
  }
}

void CalculateReference(float* next, float* prev, float* vel, float* coeff,
                        const size_t n1, const size_t n2, const size_t n3,
                        const size_t nreps) {
  for (auto it = 0; it < nreps; it += 1) {
    iso3dfdCPUIteration(next, prev, vel, coeff, n1, n2, n3);
    std::swap(next, prev);
  }
}

void VerifyResult(float* prev, float* next, float* vel, float* coeff,
                  const size_t n1, const size_t n2, const size_t n3,
                  const size_t nreps) {
  std::cout << "Running CPU version for result comparasion: ";
  auto nsize = n1 * n2 * n3;
  float* temp = new float[nsize];
  memcpy(temp, prev, nsize * sizeof(float));
  initialize(prev, next, vel, n1, n2, n3);
  CalculateReference(next, prev, vel, coeff, n1, n2, n3, nreps);
  bool error = WithinEpsilon(temp, prev, n1, n2, n3, kHalfLength, 0, 0.1f);
  if (error) {
    std::cout << "Final wavefields from SYCL device and CPU are not "
              << "equivalent: Fail\n";
  } else {
    std::cout << "Final wavefields from SYCL device and CPU are equivalent:"
              << " Success\n";
  }
  delete[] temp;
}
