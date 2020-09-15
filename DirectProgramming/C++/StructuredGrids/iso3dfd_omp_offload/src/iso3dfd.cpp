
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "../include/iso3dfd.h"

/*
 * Inline device function to find minimum
 */
#pragma omp declare target
inline unsigned int GetMin(unsigned int first, unsigned int second) {
  return ((first < second) ? first : second);
}
#pragma omp end declare target

#ifdef USE_BASELINE
/*
 * Device-Code
 * OpenMP Offload implementation for single iteration of iso3dfd kernel.
 * This function uses the default distribution of work
 * It represents minimal changes to the CPU OpenMP code.
 * Inner most loop order is changed from CPU OpenMP version to represent
 * work-items in X-Y plane. And each work-item traverses the Z-plane
 */
void inline Iso3dfdIteration(float *ptr_next_base, float *ptr_prev_base,
                             float *ptr_vel_base, float *coeff,
                             const unsigned int n1, const unsigned int n2,
                             const unsigned int n3, const unsigned int n1_block,
                             const unsigned int n2_block,
                             const unsigned int n3_block) {
  auto dimn1n2 = n1 * n2;
  auto size = n3 * dimn1n2;

  auto n3_end = n3 - kHalfLength;
  auto n2_end = n2 - kHalfLength;
  auto n1_end = n1 - kHalfLength;

  // Outer 3 loops just execute once if block sizes are same as the grid sizes,
  // which is enforced here to demonstrate the baseline version.

  for (auto bz = kHalfLength; bz < n3_end; bz += n3_block) {
    for (auto by = kHalfLength; by < n2_end; by += n2_block) {
      for (auto bx = kHalfLength; bx < n1_end; bx += n1_block) {
        auto iz_end = GetMin(bz + n3_block, n3_end);
        auto iy_end = GetMin(by + n2_block, n2_end);
        auto ix_end = GetMin(bx + n1_block, n1_end);

#pragma omp target parallel for simd collapse(3)
        for (auto iz = bz; iz < iz_end; iz++) {
          for (auto iy = by; iy < iy_end; iy++) {
            for (auto ix = bx; ix < ix_end; ix++) {
              float *ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1;
              float *ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1;
              float *ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1;

              float value = ptr_prev[ix] * coeff[0];
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
        }
      }
    }
  }
}
#endif

#ifdef USE_OPT1
/*
 * Device-Code
 * OpenMP Offload implementation for single iteration of iso3dfd kernel.
 * This function uses the tiling approach for distribution of work
 * It represents minimal changes to the CPU OpenMP code.
 * OpenMP teams are created and distributed to work on a TILE
 * Inner most loop order is changed from CPU OpenMP version to represent
 * work-items in X-Y plane. And each work-item traverses the Z-plane
 */
void inline Iso3dfdIteration(float *ptr_next_base, float *ptr_prev_base,
                             float *ptr_vel_base, float *coeff,
                             const unsigned int n1, const unsigned int n2,
                             const unsigned int n3, const unsigned int n1_block,
                             const unsigned int n2_block,
                             const unsigned int n3_block) {
  auto dimn1n2 = n1 * n2;
  auto size = n3 * dimn1n2;

  auto n3_end = n3 - kHalfLength;
  auto n2_end = n2 - kHalfLength;
  auto n1_end = n1 - kHalfLength;

#pragma omp target teams distribute collapse(3)                    \
    num_teams((n3 / n3_block) * (n2 / n2_block) * (n1 / n1_block)) \
        thread_limit(n1_block *n2_block)
  {  // start of omp target
    for (auto bz = kHalfLength; bz < n3_end; bz += n3_block) {
      for (auto by = kHalfLength; by < n2_end; by += n2_block) {
        for (auto bx = kHalfLength; bx < n1_end; bx += n1_block) {
          auto iz_end = GetMin(bz + n3_block, n3_end);
          auto iy_end = GetMin(by + n2_block, n2_end);
          auto ix_end = GetMin(bx + n1_block, n1_end);

#pragma omp parallel for simd collapse(2) schedule(static, 1)
          for (auto iy = by; iy < iy_end; iy++) {
            for (auto ix = bx; ix < ix_end; ix++) {
              for (auto iz = bz; iz < iz_end; iz++) {
                float *ptr_next = ptr_next_base + iz * dimn1n2 + iy * n1;
                float *ptr_prev = ptr_prev_base + iz * dimn1n2 + iy * n1;
                float *ptr_vel = ptr_vel_base + iz * dimn1n2 + iy * n1;

                float value = ptr_prev[ix] * coeff[0];
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
          }
        }
      }
    }
  }  // end of omp target
}
#endif

#ifdef USE_OPT2
/*
 * Device-Code
 * OpenMP Offload implementation for single iteration of iso3dfd kernel.
 * This function uses the tiling approach for distribution of work
 * It represents minimal changes to the CPU OpenMP code.
 * OpenMP teams are created and distributed to work on a TILE
 * Inner most loop order is changed from CPU OpenMP version to represent
 * work-items in X-Y plane. And each work-item traverses the Z-plane
 * In addition to this the data in the outer-most z-dimension is
 * stored locally in registers front and back for re-use
 */
void inline Iso3dfdIteration(float *ptr_next_base, float *ptr_prev_base,
                             float *ptr_vel_base, float *coeff,
                             const unsigned int n1, const unsigned int n2,
                             const unsigned int n3, const unsigned int n1_block,
                             const unsigned int n2_block,
                             const unsigned int n3_block) {
  auto dimn1n2 = n1 * n2;
  auto size = n3 * dimn1n2;

  auto n3_end = n3 - kHalfLength;
  auto n2_end = n2 - kHalfLength;
  auto n1_end = n1 - kHalfLength;
#pragma omp target teams distribute collapse(3)                    \
    num_teams((n3 / n3_block) * (n2 / n2_block) * (n1 / n1_block)) \
        thread_limit(n1_block *n2_block)
  {  // start of omp target
    for (auto bz = kHalfLength; bz < n3_end; bz += n3_block) {
      for (auto by = kHalfLength; by < n2_end; by += n2_block) {
        for (auto bx = kHalfLength; bx < n1_end; bx += n1_block) {
          auto iz_end = GetMin(bz + n3_block, n3_end);
          auto iy_end = GetMin(by + n2_block, n2_end);
          auto ix_end = GetMin(bx + n1_block, n1_end);

#pragma omp parallel for simd collapse(2) schedule(static, 1)
          for (auto iy = by; iy < iy_end; iy++) {
            for (auto ix = bx; ix < ix_end; ix++) {
              auto gid = ix + (iy * n1) + (bz * dimn1n2);
              float front[kHalfLength + 1];
              float back[kHalfLength];

              for (auto iter = 0; iter < kHalfLength; iter++) {
                front[iter] = ptr_prev_base[gid + iter * dimn1n2];
              }
              for (auto iter = 1; iter <= kHalfLength; iter++) {
                back[iter - 1] = ptr_prev_base[gid - iter * dimn1n2];
              }

              for (auto iz = bz; iz < iz_end; iz++) {
                front[kHalfLength] = ptr_prev_base[gid + kHalfLength * dimn1n2];

                float value = front[0] * coeff[0];

                value += STENCIL_LOOKUP_Z(1);
                value += STENCIL_LOOKUP_Z(2);
                value += STENCIL_LOOKUP_Z(3);
                value += STENCIL_LOOKUP_Z(4);
                value += STENCIL_LOOKUP_Z(5);
                value += STENCIL_LOOKUP_Z(6);
                value += STENCIL_LOOKUP_Z(7);
                value += STENCIL_LOOKUP_Z(8);

                ptr_next_base[gid] = 2.0f * front[0] - ptr_next_base[gid] +
                                     value * ptr_vel_base[gid];

                gid += dimn1n2;

                for (auto iter = kHalfLength - 1; iter > 0; iter--) {
                  back[iter] = back[iter - 1];
                }
                back[0] = front[0];

                for (auto iter = 0; iter < kHalfLength; iter++) {
                  front[iter] = front[iter + 1];
                }
              }
            }
          }
        }
      }
    }
  }  // end of omp target
}
#endif

#ifdef USE_OPT3
/*
 * Device-Code
 * OpenMP Offload implementation for single iteration of iso3dfd kernel.
 * In this version the 3D-stencil is decomposed into smaller grids
 * along the outer-most z-dimension. This results in multiple openmp CPU
 * threads invoking omp target device kernels.
 * This version also uses the tiling approach for distribution of work
 * It represents minimal changes to the CPU OpenMP code.
 * OpenMP teams are created and distributed to work on a TILE
 * Inner most loop order is changed from CPU OpenMP version to represent
 * work-items in X-Y plane. And each work-item traverses the Z-plane
 * In addition to this the data in the outer-most z-dimension is
 * stored locally in registers front and back for re-use
 */
void inline Iso3dfdIteration(float *ptr_next_base, float *ptr_prev_base,
                             float *ptr_vel_base, float *coeff,
                             const unsigned int n1, const unsigned int n2,
                             const unsigned int n3, const unsigned int n1_block,
                             const unsigned int n2_block,
                             const unsigned int n3_block) {
  auto dimn1n2 = n1 * n2;
  auto size = n3 * dimn1n2;

  auto n3_end = n3 - kHalfLength;
  auto n2_end = n2 - kHalfLength;
  auto n1_end = n1 - kHalfLength;
#pragma omp parallel for
  for (auto bz = kHalfLength; bz < n3_end; bz += n3_block) {
#pragma omp target teams distribute collapse(2)  \
    num_teams((n2 / n2_block) * (n1 / n1_block)) \
        thread_limit(n1_block *n2_block)
    for (auto by = kHalfLength; by < n2_end; by += n2_block) {
      for (auto bx = kHalfLength; bx < n1_end; bx += n1_block) {
        auto iz_end = GetMin(bz + n3_block, n3_end);
        auto iy_end = GetMin(by + n2_block, n2_end);
        auto ix_end = GetMin(bx + n1_block, n1_end);

#pragma omp parallel for simd collapse(2) schedule(static, 1)
        for (auto iy = by; iy < iy_end; iy++) {
          for (auto ix = bx; ix < ix_end; ix++) {
            auto gid = ix + (iy * n1) + (bz * dimn1n2);
            float front[kHalfLength + 1];
            float back[kHalfLength];

            for (auto iter = 0; iter < kHalfLength; iter++) {
              front[iter] = ptr_prev_base[gid + iter * dimn1n2];
            }
            for (auto iter = 1; iter <= kHalfLength; iter++) {
              back[iter - 1] = ptr_prev_base[gid - iter * dimn1n2];
            }

            for (auto iz = bz; iz < iz_end; iz++) {
              front[kHalfLength] = ptr_prev_base[gid + kHalfLength * dimn1n2];

              float value = front[0] * coeff[0];

              value += STENCIL_LOOKUP_Z(1);
              value += STENCIL_LOOKUP_Z(2);
              value += STENCIL_LOOKUP_Z(3);
              value += STENCIL_LOOKUP_Z(4);
              value += STENCIL_LOOKUP_Z(5);
              value += STENCIL_LOOKUP_Z(6);
              value += STENCIL_LOOKUP_Z(7);
              value += STENCIL_LOOKUP_Z(8);

              ptr_next_base[gid] = 2.0f * front[0] - ptr_next_base[gid] +
                                   value * ptr_vel_base[gid];

              gid += dimn1n2;

              for (auto iter = kHalfLength - 1; iter > 0; iter--) {
                back[iter] = back[iter - 1];
              }
              back[0] = front[0];

              for (auto iter = 0; iter < kHalfLength; iter++) {
                front[iter] = front[iter + 1];
              }
            }
          }
        }
      }
    }
  }
}
#endif

/*
 * Host-Code
 * Driver function for ISO3DFD OpenMP Offload code
 * Uses ptr_next and ptr_prev as ping-pong buffers to achieve
 * accelerated wave propogation
 * OpenMP Target region is declared and maintainted for all the
 * time steps
 */
void Iso3dfd(float *ptr_next, float *ptr_prev, float *ptr_vel, float *coeff,
             const unsigned int n1, const unsigned int n2,
             const unsigned int n3, const unsigned int nreps,
             const unsigned int n1_block, const unsigned int n2_block,
             const unsigned int n3_block) {
  auto dimn1n2 = n1 * n2;
  auto size = n3 * dimn1n2;

  float *temp = NULL;

#pragma omp target data map(ptr_next [0:size], ptr_prev [0:size]) map( \
    ptr_vel [0:size], coeff [0:9], n1, n2, n3, n1_block, n2_block, n3_block)
  for (auto it = 0; it < nreps; it += 1) {
#ifdef USE_BASELINE
    Iso3dfdIteration(ptr_next, ptr_prev, ptr_vel, coeff, n1, n2, n3, n1, n2,
                     n3);
#else
    Iso3dfdIteration(ptr_next, ptr_prev, ptr_vel, coeff, n1, n2, n3, n1_block,
                     n2_block, n3_block);
#endif
    // here's where boundary conditions and halo exchanges happen
    temp = ptr_next;
    ptr_next = ptr_prev;
    ptr_prev = temp;
  }
}

int main(int argc, char *argv[]) {
  // Arrays used to update the wavefield
  float *prev_base;
  float *next_base;
  // Array to store wave velocity
  float *vel_base;

  bool error = false;

  unsigned int n1, n2, n3;
  unsigned int n1_block, n2_block, n3_block;
  unsigned int num_iterations;

  try {
    n1 = std::stoi(argv[1]) + (2 * kHalfLength);
    n2 = std::stoi(argv[2]) + (2 * kHalfLength);
    n3 = std::stoi(argv[3]) + (2 * kHalfLength);
    n1_block = std::stoi(argv[4]);
    n2_block = std::stoi(argv[5]);
    n3_block = std::stoi(argv[6]);
    num_iterations = std::stoi(argv[7]);
  }

  catch (...) {
    Usage(argv[0]);
    return 1;
  }

  if (ValidateInput(std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]),
                    n1_block, n2_block, n3_block, num_iterations)) {
    Usage(argv[0]);
    return 1;
  }

  // Check for available omp offload capable device
  unsigned int num_devices = omp_get_num_devices();
  if (num_devices <= 0) {
    std::cout << "--------------------------------------\n";
    std::cout << " No OpenMP Offload device found\n";
    Usage(argv[0]);
    return 1;
  }

  auto nsize = n1 * n2 * n3;

  prev_base = new float[nsize];
  next_base = new float[nsize];
  vel_base = new float[nsize];

  // Compute coefficients to be used in wavefield update
  float coeff[kHalfLength + 1] = {-3.0548446,   +1.7777778,     -3.1111111e-1,
                                  +7.572087e-2, -1.76767677e-2, +3.480962e-3,
                                  -5.180005e-4, +5.074287e-5,   -2.42812e-6};

  // Apply the DX DY and DZ to coefficients
  coeff[0] = (3.0f * coeff[0]) / (dxyz * dxyz);
  for (auto i = 1; i <= kHalfLength; i++) {
    coeff[i] = coeff[i] / (dxyz * dxyz);
  }

  Initialize(prev_base, next_base, vel_base, n1, n2, n3);

  std::cout << "Grid Sizes: " << n1 - 2 * kHalfLength << " "
            << n2 - 2 * kHalfLength << " " << n3 - 2 * kHalfLength << "\n";
#if defined(USE_OPT1) || defined(USE_OPT2) || defined(USE_OPT3)
  std::cout << "Tile sizes: " << n1_block << " " << n2_block << " " << n3_block
            << "\n";
#ifdef USE_OPT1
  std::cout << "Using Optimized target code - version 1:\n";
  std::cout << "--OMP_Offload with Tiling\n";
#elif USE_OPT2
  std::cout << "Using Optimized target code - version 2:\n";
  std::cout << "--OMP_Offload with Tiling and Z Window\n";
#elif USE_OPT3
  std::cout << "Using Optimized target code - version 3:\n";
  std::cout << "--OMP Threads + OMP_Offload with Tiling and Z Window\n";
#endif
#else
  std::cout << "Tile sizes ignored for OMP Offload\n";
  std::cout << "--Using Baseline version with omp target with collapse\n";
#endif
  std::cout << "Memory Usage (MBytes): "
            << ((3 * nsize * sizeof(float)) / (1024 * 1024)) << "\n";

  auto start = std::chrono::steady_clock::now();

  Iso3dfd(next_base, prev_base, vel_base, coeff, n1, n2, n3, num_iterations,
          n1_block, n2_block, n3_block);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

  PrintStats(time, n1, n2, n3, num_iterations);

#ifdef VERIFY_RESULTS
  error = VerifyResults(next_base, prev_base, vel_base, coeff, n1, n2, n3,
                        num_iterations, n1_block, n2_block, n3_block);
#endif
  delete[] prev_base;
  delete[] next_base;
  delete[] vel_base;

  return error ? 1 : 0;
}
