//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "iso3dfd.h"

/*
 * Host-Code
 * Utility function to validate grid and block dimensions
 */
bool CheckGridDimension(size_t n1, size_t n2, size_t n3, unsigned int dim_x,
                        unsigned int dim_y, unsigned int block_z) {
  if (n1 % dim_x) {
    std::cout << " ERROR: Invalid Grid Size: n1 should be multiple of DIMX - "
              << dim_x << "\n";
    return true;
  }
  if (n2 % dim_y) {
    std::cout << " ERROR: Invalid Grid Size: n2 should be multiple of DIMY - "
              << dim_y << "\n";
    return true;
  }
  if (n3 % block_z) {
    std::cout << " ERROR: Invalid Grid Size: n3 should be multiple of BLOCKZ - "
              << block_z << "\n";
    return true;
  }

  return false;
}

/*
 * Host-Code
 * Utility function to validate block sizes
 */
bool CheckBlockDimension(sycl::queue& q, unsigned int dim_x,
                         unsigned int dim_y) {
  auto device = q.get_device();
  auto max_block_size =
      device.get_info<sycl::info::device::max_work_group_size>();

  if ((max_block_size > 1) && (dim_x * dim_y > max_block_size)) {
    std::cout << "ERROR: Invalid block sizes: n1_Tblock * n2_Tblock should be "
                 "less than or equal to "
              << max_block_size << "\n";
    return true;
  }

  return false;
}

/*
 * Host-Code
 * Utility function to print device info
 */
void PrintTargetInfo(sycl::queue& q, unsigned int dim_x, unsigned int dim_y) {
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
  std::cout << " The blockSize x is : " << dim_x << "\n";
  std::cout << " The blockSize y is : " << dim_y << "\n";
#ifdef USE_SHARED
  std::cout << " Using Shared Local Memory Kernel\n";
#else
  std::cout << " Using Global Memory Kernel\n";

#endif
}

/*
 * Host-Code
 * Utility function to get input arguments
 */
void Usage(const std::string& programName) {
  std::cout << " Incorrect parameters \n";
  std::cout << " Usage: ";
  std::cout << programName
            << " n1 n2 n3 b1 b2 b3 Iterations [omp|sycl] [gpu|cpu] \n\n";
  std::cout << " n1 n2 n3      : Grid sizes for the stencil \n";
  std::cout << " b1 b2 b3      : cache block sizes for cpu openmp version.\n";
  std::cout << " Iterations    : No. of timesteps. \n";
  std::cout << " [omp|sycl]    : Optional: Run the OpenMP or the SYCL variant."
            << " Default is to use both for validation \n";
  std::cout
      << " [gpu|cpu]     : Optional: Device to run the SYCL version"
      << " Default is to use the GPU if available, if not fallback to CPU \n\n";
}

/*
 * Host-Code
 * Utility function to print stats
 */
void PrintStats(double time, size_t n1, size_t n2, size_t n3,
                unsigned int nIterations) {
  float throughput_mpoints = 0.0f, mflops = 0.0f, normalized_time = 0.0f;
  double mbytes = 0.0f;

  normalized_time = (double)time / nIterations;
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
