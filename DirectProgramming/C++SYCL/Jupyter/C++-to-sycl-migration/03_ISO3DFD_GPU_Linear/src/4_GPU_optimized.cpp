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

void iso3dfd(queue& q, float* ptr_next, float* ptr_prev, float* ptr_vel, float* ptr_coeff,
             const size_t n1, const size_t n2, const size_t n3,size_t n1_block, size_t n2_block, size_t n3_block,
             const size_t nIterations) {
  auto nx = n1;
  auto nxy = n1*n2;
  auto grid_size = nxy*n3;

  auto b1 = kHalfLength;
  auto b2 = kHalfLength;
  auto b3 = kHalfLength;

  auto next = sycl::aligned_alloc_device<float>(64, grid_size + 16, q);
  next += (16 - b1);
  q.memcpy(next, ptr_next, sizeof(float)*grid_size);
  auto prev = sycl::aligned_alloc_device<float>(64, grid_size + 16, q);
  prev += (16 - b1);
  q.memcpy(prev, ptr_prev, sizeof(float)*grid_size);
  auto vel = sycl::aligned_alloc_device<float>(64, grid_size + 16, q);
  vel += (16 - b1);
  q.memcpy(vel, ptr_vel, sizeof(float)*grid_size);
  //auto coeff = sycl::aligned_alloc_device<float>(64, grid_size + 16, q);
  auto coeff = sycl::aligned_alloc_device<float>(64, kHalfLength+1 , q);
  q.memcpy(coeff, ptr_coeff, sizeof(float)*(kHalfLength+1));
  //coeff += (16 - b1);
  //q.memcpy(coeff, coeff, sizeof(float)*grid_size);
  q.wait();

  //auto local_nd_range = range(1, n2_block, n1_block);
  //auto global_nd_range = range((n3 - 2 * kHalfLength)/n3_block, (n2 - 2 * kHalfLength)/n2_block,
                  //(n1 - 2 * kHalfLength));
				  
  auto local_nd_range = range<3>(n3_block,n2_block,n1_block);
  auto global_nd_range = range<3>((n3-2*b3+n3_block-1)/n3_block*n3_block,(n2-2*b2+n2_block-1)/n2_block*n2_block,n1_block);
  

  for (auto i = 0; i < nIterations; i += 1) {
    q.submit([&](auto &h) {      
        h.parallel_for(
              nd_range(global_nd_range, local_nd_range), [=](auto item)
          //[[intel::reqd_sub_group_size(32)]]
          //[[intel::kernel_args_restrict]]
         {
            const int iz = b3 + item.get_global_id(0);
            const int iy = b2 + item.get_global_id(1);
            if (iz < n3 - b3 && iy < n2 - b2)
             for (int ix = b1+item.get_global_id(2); ix < n1 - b1; ix += n1_block)
                {
                  auto gid = ix + iy*nx + iz*nxy;
                  float *pgid = prev+gid;
                  auto value = coeff[0] * pgid[0];
#pragma unroll(kHalfLength)
                  for (auto iter = 1; iter <= kHalfLength; iter++)
                    value += coeff[iter]*(pgid[iter*nxy] + pgid[-iter*nxy] + pgid[iter*nx] + pgid[-iter*nx] + pgid[iter] + pgid[-iter]);
                  next[gid] = 2.0f*pgid[0] - next[gid] + value*vel[gid];
                }
      });    
    }).wait();
   std::swap(next, prev);
  }
  q.memcpy(ptr_prev, prev, sizeof(float)*grid_size);

  sycl::free(next - (16 - b1),q);
  sycl::free(prev - (16 - b1),q);
  sycl::free(vel - (16 - b1),q);
  sycl::free(coeff,q);  

}

int main(int argc, char* argv[]) {
  // Arrays used to update the wavefield
  float* prev;
  float* next;
  // Array to store wave velocity
  float* vel;

  // Variables to store size of grids and number of simulation iterations
  size_t n1, n2, n3;
    size_t n1_block, n2_block, n3_block;
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
    n1_block = std::stoi(argv[4]);
    n2_block = std::stoi(argv[5]);
    n3_block = std::stoi(argv[6]);
    num_iterations = std::stoi(argv[7]);    
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
  iso3dfd(q, next, prev, vel, coeff, n1, n2, n3,n1_block,n2_block,n3_block, num_iterations);

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
