//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "../include/iso3dfd.h"

/*
 * Host-Code
 * Utility function to get input arguments
 */
void Usage(const std::string& programName) {
  std::cout << "--------------------------------------\n";
  std::cout << " Incorrect parameters \n";
  std::cout << " Usage: ";
  std::cout << programName
            << " n1 n2 n3 n1_block n2_block n3_block Iterations\n\n";
  std::cout << " n1 n2 n3      			: Grid sizes for the stencil\n";
  std::cout << " n1_block n2_block n3_block     : cache block sizes for CPU\n";
  std::cout << " 	       			: TILE sizes for OMP Offload\n";
  std::cout << " Iterations    			: No. of timesteps.\n";
  std::cout << "--------------------------------------\n";
  std::cout << "--------------------------------------\n";
}

/*
 * Host-Code
 * Function used for initialization
 */
void Initialize(float* ptr_prev, float* ptr_next, float* ptr_vel,
                unsigned int n1, unsigned int n2, unsigned int n3) {
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

/*
 * Host-Code
 * Utility function to print stats
 */
void PrintStats(double time, unsigned int n1, unsigned int n2, unsigned int n3,
                unsigned int num_iterations) {
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

/*
 * Host-Code
 * Utility function to calculate L2-norm between resulting buffer and reference
 * buffer
 */
bool WithinEpsilon(float* output, float* reference, unsigned int dim_x,
                   unsigned int dim_y, unsigned int dim_z, unsigned int radius,
                   const int zadjust = 0, const float delta = 0.01f) {
  std::ofstream error_file;
  error_file.open("error_diff.txt");

  bool error = false;
  double norm2 = 0;

  for (auto iz = 0; iz < dim_z; iz++) {
    for (auto iy = 0; iy < dim_y; iy++) {
      for (auto ix = 0; ix < dim_x; ix++) {
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

/*
 * Host-code
 * Validate input arguments
 */
bool ValidateInput(unsigned int n1, unsigned int n2, unsigned int n3,
                   unsigned int n1_block, unsigned int n2_block,
                   unsigned int n3_block, unsigned int num_iterations) {
  bool error = false;

  if ((n1 < kHalfLength) || (n2 < kHalfLength) || (n3 < kHalfLength)) {
    std::cout << "--------------------------------------\n";
    std::cout << " Invalid grid size : n1, n2, n3 should be greater than "
              << kHalfLength << "\n";
    error = true;
  }
  if ((n1_block <= 0) || (n2_block <= 0) || (n3_block <= 0)) {
    std::cout << "--------------------------------------\n";
    std::cout << " Invalid block sizes : n1_block, n2_block, n3_block "
                 "should be greater than 0\n";
    error = true;
  }
  if (num_iterations <= 0) {
    std::cout << "--------------------------------------\n";
    std::cout
        << " Invalid num_iterations :  Iterations should be greater than 0 \n";
    error = true;
  }

#if defined(USE_OPT1) || defined(USE_OPT2) || defined(USE_OPT3)
  if ((n1_block * n2_block) > kMaxTeamSizeLimit) {
    std::cout << "--------------------------------------\n";
    std::cout << " Invalid block sizes : n1_block * n2_block "
                 "should be less than "
              << kMaxTeamSizeLimit << "\n";
    error = true;
  }
#endif

  return error;
}
