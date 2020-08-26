//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "../include/iso3dfd.h"

/*
 * Host-Code
 * OpenMP implementation for single iteration of iso3dfd kernel.
 * This function is used as reference implementation for verification and
 * also to compare OpenMP performance on CPU with the OpenMP Offload version
 */
void Iso3dfdVerifyIteration(float *ptr_next_base, float *ptr_prev_base,
                            float *ptr_vel_base, float *coeff, int n1, int n2,
                            int n3, unsigned int n1_block,
                            unsigned int n2_block, unsigned int n3_block) {
  auto dimn1n2 = n1 * n2;

  auto n3_end = n3 - kHalfLength;
  auto n2_end = n2 - kHalfLength;
  auto n1_end = n1 - kHalfLength;

#pragma omp parallel default(shared)
#pragma omp for schedule(static) collapse(3)
  for (auto bz = kHalfLength; bz < n3_end; bz += n3_block) {
    for (auto by = kHalfLength; by < n2_end; by += n2_block) {
      for (auto bx = kHalfLength; bx < n1_end; bx += n1_block) {
        auto iz_end = std::min(bz + n3_block, n3_end);
        auto iy_end = std::min(by + n2_block, n2_end);
        auto ix_end = std::min(n1_block, n1_end - bx);
        for (auto iz = bz; iz < iz_end; iz++) {
          for (auto iy = by; iy < iy_end; iy++) {
            float *ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1 + bx;
            float *ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1 + bx;
            float *ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1 + bx;
#pragma omp simd
            for (auto ix = 0; ix < ix_end; ix++) {
              float value = 0.0f;
              value += ptr_prev[ix] * coeff[0];
              value += STENCIL_LOOKUP(1);
              value += STENCIL_LOOKUP(2);
              value += STENCIL_LOOKUP(3);
              value += STENCIL_LOOKUP(4);
              value += STENCIL_LOOKUP(5);
              value += STENCIL_LOOKUP(6);
              value += STENCIL_LOOKUP(7);
              value += STENCIL_LOOKUP(8);
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
 * Driver function for ISO3DFD OpenMP CPU code
 * Uses ptr_next and ptr_prev as ping-pong buffers to achieve
 * accelerated wave propogation
 */
void Iso3dfdVerify(float *ptr_next, float *ptr_prev, float *ptr_vel,
                   float *coeff, unsigned int n1, unsigned int n2,
                   unsigned int n3, unsigned int nreps, unsigned int n1_block,
                   unsigned int n2_block, unsigned int n3_block) {
  for (auto it = 0; it < nreps; it += 1) {
    Iso3dfdVerifyIteration(ptr_next, ptr_prev, ptr_vel, coeff, n1, n2, n3,
                           n1_block, n2_block, n3_block);

    // here's where boundary conditions and halo exchanges happen
    // Swap previous & next between iterations
    it++;
    if (it < nreps)
      Iso3dfdVerifyIteration(ptr_prev, ptr_next, ptr_vel, coeff, n1, n2, n3,
                             n1_block, n2_block, n3_block);

  }  // time loop
}

bool VerifyResults(float *next_base, float *prev_base, float *vel_base,
                   float *coeff, unsigned int n1, unsigned int n2,
                   unsigned int n3, unsigned int num_iterations,
                   unsigned int n1_block, unsigned int n2_block,
                   unsigned int n3_block) {
  std::cout << "Checking Results ...\n";
  size_t nsize = n1 * n2 * n3;
  bool error = false;

  float *temp = new float[nsize];
  if (num_iterations % 2)
    memcpy(temp, next_base, nsize * sizeof(float));
  else
    memcpy(temp, prev_base, nsize * sizeof(float));

  Initialize(prev_base, next_base, vel_base, n1, n2, n3);

  Iso3dfdVerify(next_base, prev_base, vel_base, coeff, n1, n2, n3,
                num_iterations, n1_block, n2_block, n3_block);

  if (num_iterations % 2)
    error = WithinEpsilon(temp, next_base, n1, n2, n3, kHalfLength, 0, 0.1f);
  else
    error = WithinEpsilon(temp, prev_base, n1, n2, n3, kHalfLength, 0, 0.1f);

  if (error) {
    std::cout << "Final wavefields from OMP Offload device and CPU are not "
              << "equivalent: Fail\n";
  } else {
    std::cout << "Final wavefields from OMP Offload device and CPU are "
              << "equivalent: Success\n";
  }
  std::cout << "--------------------------------------\n";
  delete[] temp;

  return error;
}
