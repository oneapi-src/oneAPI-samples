//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "../include/iso3dfd.h"

/*
 * Host-Code
 * Utility function to validate grid and block dimensions
 */
bool checkGridDimension(size_t n1, size_t n2, size_t n3, unsigned int dimX,
                        unsigned int dimY, unsigned int blockZ) {
  if (n1 % dimX) {
    std::cout << " ERROR: Invalid Grid Size: n1 should be multiple of DIMX - "
              << dimX << "\n";
    return true;
  }
  if (n2 % dimY) {
    std::cout << " ERROR: Invalid Grid Size: n2 should be multiple of DIMY - "
              << dimY << "\n";
    ;
    return true;
  }
  if (n3 % blockZ) {
    std::cout << " ERROR: Invalid Grid Size: n3 should be multiple of BLOCKZ - "
              << blockZ << "\n";
    ;
    return true;
  }

  return false;
}

/*
 * Host-Code
 * Utility function to validate block sizes
 */
bool checkBlockDimension(cl::sycl::queue& q, unsigned int dimX,
                         unsigned int dimY) {
  auto device = q.get_device();
  auto maxBlockSize =
      device.get_info<cl::sycl::info::device::max_work_group_size>();

  if ((maxBlockSize > 1) && (dimX * dimY > maxBlockSize)) {
    std::cout << "ERROR: Invalid block sizes: n1_Tblock * n2_Tblock should be "
                 "less than or equal to "
              << maxBlockSize << "\n";
    ;
    return true;
  }

  return false;
}

/*
 * Host-Code
 * Utility function to print device info
 */
void printTargetInfo(cl::sycl::queue& q, unsigned int dimX, unsigned int dimY) {
  auto device = q.get_device();
  auto maxBlockSize =
      device.get_info<cl::sycl::info::device::max_work_group_size>();

  auto maxEUCount =
      device.get_info<cl::sycl::info::device::max_compute_units>();

  std::cout << " Running on " << device.get_info<cl::sycl::info::device::name>()
            << "\n";
  std::cout << " The Device Max Work Group Size is : " << maxBlockSize
            << "\n";
  std::cout << " The Device Max EUCount is : " << maxEUCount << "\n";
  std::cout << " The blockSize x is : " << dimX << "\n";
  std::cout << " The blockSize y is : " << dimY << "\n";
#ifdef USE_SHARED
  std::cout << " Using Shared Local Memory Kernel : " << "\n";
#else
  std::cout << " Using Global Memory Kernel : " << "\n";

#endif
}

/*
 * Host-Code
 * Utility function to get input arguments
 */
void usage(std::string programName) {
  std::cout << " Incorrect parameters " << "\n";
  std::cout << " Usage: ";
  std::cout << programName
            << " n1 n2 n3 b1 b2 b3 Iterations [omp|sycl] [gpu|cpu]" << "\n"
            << "\n";
  std::cout << " n1 n2 n3      : Grid sizes for the stencil " << "\n";
  std::cout << " b1 b2 b3      : cache block sizes for cpu openmp version. "
            << "\n";
  std::cout << " Iterations    : No. of timesteps. " << "\n";
  std::cout << " [omp|sycl]    : Optional: Run the OpenMP or the SYCL variant."
            << " Default is to use both for validation " << "\n";
  std::cout
      << " [gpu|cpu]     : Optional: Device to run the SYCL version"
      << " Default is to use the GPU if available, if not fallback to CPU "
      << "\n"
      << "\n";
}

/*
 * Host-Code
 * Utility function to print stats
 */
void printStats(double time, size_t n1, size_t n2, size_t n3,
                unsigned int nIterations) {
  float throughput_mpoints = 0.0f, mflops = 0.0f, normalized_time = 0.0f;
  double mbytes = 0.0f;

  normalized_time = (double)time / nIterations;
  throughput_mpoints = ((n1 - 2 * HALF_LENGTH) * (n2 - 2 * HALF_LENGTH) *
                        (n3 - 2 * HALF_LENGTH)) /
                       (normalized_time * 1e3f);
  mflops = (7.0f * HALF_LENGTH + 5.0f) * throughput_mpoints;
  mbytes = 12.0f * throughput_mpoints;

  std::cout << "--------------------------------------" << "\n";
  std::cout << "time         : " << time / 1e3f << " secs" << "\n";
  std::cout << "throughput   : " << throughput_mpoints << " Mpts/s"
            << "\n";
  std::cout << "flops        : " << mflops / 1e3f << " GFlops" << "\n";
  std::cout << "bytes        : " << mbytes / 1e3f << " GBytes/s" << "\n";
  std::cout << "\n"
            << "--------------------------------------" << "\n";
  std::cout << "\n"
            << "--------------------------------------" << "\n";
}

/*
 * Host-Code
 * Utility function to calculate L2-norm between resulting buffer and reference
 * buffer
 */
bool within_epsilon(float* output, float* reference, const size_t dimx,
                    const size_t dimy, const size_t dimz,
                    const unsigned int radius, const int zadjust = 0,
                    const float delta = 0.01f) {
  FILE* fp = fopen("./error_diff.txt", "w");
  if (!fp) fp = stderr;

  bool error = false;
  double norm2 = 0;

  for (size_t iz = 0; iz < dimz; iz++) {
    for (size_t iy = 0; iy < dimy; iy++) {
      for (size_t ix = 0; ix < dimx; ix++) {
        if (ix >= radius && ix < (dimx - radius) && iy >= radius &&
            iy < (dimy - radius) && iz >= radius &&
            iz < (dimz - radius + zadjust)) {
          float difference = fabsf(*reference - *output);
          norm2 += difference * difference;
          if (difference > delta) {
            error = true;
            fprintf(fp, " ERROR: (%zu,%zu,%zu)\t%e instead of %e (|e|=%e)\n",
                    ix, iy, iz, *output, *reference, difference);
          }
        }
        ++output;
        ++reference;
      }
    }
  }

  if (fp != stderr) fclose(fp);
  norm2 = sqrt(norm2);
  if (error) printf("error (Euclidean norm): %.9e\n", norm2);
  return error;
}

