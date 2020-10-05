//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// ISO3DFD: Intel® oneAPI DPC++ Language Basics Using 3D-Finite-Difference-Wave
// Propagation
//
// ISO3DFD is a finite difference stencil kernel for solving the 3D acoustic
// isotropic wave equation which can be used as a proxy for propogating a
// seismic wave. Kernels in this sample are implemented as 16th order in space,
// with symmetric coefficients, and 2nd order in time scheme without boundary
// conditions.. Using Data Parallel C++, the sample can explicitly run on the
// GPU and/or CPU to propagate a seismic wave which is a compute intensive task.
// If successful, the output will print the device name
// where the DPC++ code ran along with the grid computation metrics - flops
// and effective throughput.
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide
// and search based on relevant terms noted in the comments.
//
// DPC++ material used in this code sample:
//
// DPC++ Queues (including device selectors and exception handlers)
// DPC++ Custom device selector
// DPC++ Buffers and accessors (communicate data between the host and the
// device)
// DPC++ Kernels (including parallel_for function and nd-range<3>
// objects)
// Shared Local Memory (SLM) optimizations (DPC++)
// DPC++ Basic synchronization (barrier function)
//
#include "iso3dfd.h"

/*
 * Device-Code - Optimized for GPU
 * SYCL implementation for single iteration of iso3dfd kernel
 * using shared local memory optimizations
 *
 * ND-Range kernel is used to spawn work-items in x, y dimension
 * Each work-item then traverses in the z-dimension
 *
 * z-dimension slicing can be used to vary the total number
 * global work-items.
 *
 * SLM Padding can be used to eliminate SLM bank conflicts if
 * there are any
 */
void Iso3dfdIterationSLM(sycl::nd_item<3> it, float *next, float *prev,
                         float *vel, const float *coeff, float *tab, size_t nx,
                         size_t nxy, size_t bx, size_t by, size_t z_offset,
                         int full_end_z) {
  // Compute local-id for each work-item
  auto id0 = it.get_local_id(2);
  auto id1 = it.get_local_id(1);

  // Compute the position in local memory each work-item
  // will fetch data from global memory into shared
  // local memory
  auto stride = it.get_local_range(2) + 2 * kHalfLength + kPad;
  auto identifiant = (id0 + kHalfLength) + (id1 + kHalfLength) * stride;

  // We compute the start and the end position in the grid
  // for each work-item.
  // Each work-items local value gid is updated to track the
  // current cell/grid point it is working with.
  // This position is calculated with the help of slice-ID and number of
  // grid points each work-item will process.
  // Offset of kHalfLength is also used to account for HALO
  auto begin_z = it.get_global_id(0) * z_offset + kHalfLength;
  auto end_z = begin_z + z_offset;
  if (end_z > full_end_z) end_z = full_end_z;

  auto gid = (it.get_global_id(2) + bx) + ((it.get_global_id(1) + by) * nx) +
             (begin_z * nxy);

  // front and back temporary arrays are used to ensure
  // the grid values in z-dimension are read once, shifted in
  // these array and re-used multiple times before being discarded
  //
  // This is an optimization technique to enable data-reuse and
  // improve overall FLOPS to BYTES read ratio
  float front[kHalfLength + 1];
  float back[kHalfLength];
  float c[kHalfLength + 1];

  for (auto iter = 0; iter < kHalfLength; iter++) {
    front[iter] = prev[gid + iter * nxy];
  }
  c[0] = coeff[0];

  for (auto iter = 1; iter <= kHalfLength; iter++) {
    back[iter - 1] = prev[gid - iter * nxy];
    c[iter] = coeff[iter];
  }

  // Shared Local Memory (SLM) optimizations (DPC++)
  // Set some flags to indicate if the current work-item
  // should read from global memory to shared local memory buffer
  // or not
  auto items_x = it.get_local_range(2);
  auto items_y = it.get_local_range(1);

  bool copy_halo_y = false, copy_halo_x = false;
  if (id1 < kHalfLength) copy_halo_y = true;
  if (id0 < kHalfLength) copy_halo_x = true;

  for (auto i = begin_z; i < end_z; i++) {
    // Shared Local Memory (SLM) optimizations (DPC++)
    // If work-item is flagged to read into SLM buffer
    if (copy_halo_y) {
      tab[identifiant - kHalfLength * stride] = prev[gid - kHalfLength * nx];
      tab[identifiant + items_y * stride] = prev[gid + items_y * nx];
    }
    if (copy_halo_x) {
      tab[identifiant - kHalfLength] = prev[gid - kHalfLength];
      tab[identifiant + items_x] = prev[gid + items_x];
    }
    tab[identifiant] = front[0];

    // DPC++ Basic synchronization (barrier function)
    // Force synchronization within a work-group
    // using barrier function to ensure
    // all the work-items have completed reading into the SLM buffer
    it.barrier(access::fence_space::local_space);

    // Only one new data-point read from global memory
    // in z-dimension (depth)
    front[kHalfLength] = prev[gid + kHalfLength * nxy];

    // Stencil code to update grid point at position given by global id (gid)
    // New time step for grid point is computed based on the values of the
    // the immediate neighbors - horizontal, vertical and depth
    // directions(kHalfLength number of points in each direction),
    // as well as the value of grid point at a previous time step
    //
    // Neighbors in the depth (z-dimension) are read out of
    // front and back arrays
    // Neighbors in the horizontal and vertical (x, y dimension) are
    // read from the SLM buffers
    float value = c[0] * front[0];
#pragma unroll(kHalfLength)
    for (auto iter = 1; iter <= kHalfLength; iter++) {
      value += c[iter] *
               (front[iter] + back[iter - 1] + tab[identifiant + iter] +
                tab[identifiant - iter] + tab[identifiant + iter * stride] +
                tab[identifiant - iter * stride]);
    }
    next[gid] = 2.0f * front[0] - next[gid] + value * vel[gid];

    // Update the gid to advance in the z-dimension
    gid += nxy;

    // Input data in front and back are shifted to discard the
    // oldest value and read one new value.
    for (auto iter = kHalfLength - 1; iter > 0; iter--) {
      back[iter] = back[iter - 1];
    }
    back[0] = front[0];

    for (auto iter = 0; iter < kHalfLength; iter++) {
      front[iter] = front[iter + 1];
    }

    // DPC++ Basic synchronization (barrier function)
    // Force synchronization within a work-group
    // using barrier function to ensure that SLM buffers
    // are not overwritten by next set of work-items
    // (highly unlikely but not impossible)
    it.barrier(access::fence_space::local_space);
  }
}

/*
 * Device-Code - Optimized for GPU, CPU
 * SYCL implementation for single iteration of iso3dfd kernel
 * without using any shared local memory optimizations
 *
 *
 * ND-Range kernel is used to spawn work-items in x, y dimension
 * Each work-item can then traverse in the z-dimension
 *
 * z-dimension slicing can be used to vary the total number
 * global work-items.
 *
 */
void Iso3dfdIterationGlobal(sycl::nd_item<3> it, float *next, float *prev,
                            float *vel, const float *coeff, int nx, int nxy,
                            int bx, int by, int z_offset, int full_end_z) {
  // We compute the start and the end position in the grid
  // for each work-item.
  // Each work-items local value gid is updated to track the
  // current cell/grid point it is working with.
  // This position is calculated with the help of slice-ID and number of
  // grid points each work-item will process.
  // Offset of kHalfLength is also used to account for HALO
  auto begin_z = it.get_global_id(0) * z_offset + kHalfLength;
  auto end_z = begin_z + z_offset;
  if (end_z > full_end_z) end_z = full_end_z;

  auto gid = (it.get_global_id(2) + bx) + ((it.get_global_id(1) + by) * nx) +
             (begin_z * nxy);

  // front and back temporary arrays are used to ensure
  // the grid values in z-dimension are read once, shifted in
  // these array and re-used multiple times before being discarded
  //
  // This is an optimization technique to enable data-reuse and
  // improve overall FLOPS to BYTES read ratio
  float front[kHalfLength + 1];
  float back[kHalfLength];
  float c[kHalfLength + 1];

  for (auto iter = 0; iter <= kHalfLength; iter++) {
    front[iter] = prev[gid + iter * nxy];
  }
  c[0] = coeff[0];
  for (auto iter = 1; iter <= kHalfLength; iter++) {
    c[iter] = coeff[iter];
    back[iter - 1] = prev[gid - iter * nxy];
  }

  // Stencil code to update grid point at position given by global id (gid)
  // New time step for grid point is computed based on the values of the
  // the immediate neighbors - horizontal, vertical and depth
  // directions(kHalfLength number of points in each direction),
  // as well as the value of grid point at a previous time step

  float value = c[0] * front[0];
#pragma unroll(kHalfLength)
  for (auto iter = 1; iter <= kHalfLength; iter++) {
    value += c[iter] *
             (front[iter] + back[iter - 1] + prev[gid + iter] +
              prev[gid - iter] + prev[gid + iter * nx] + prev[gid - iter * nx]);
  }
  next[gid] = 2.0f * front[0] - next[gid] + value * vel[gid];

  // Update the gid and position in z-dimension and check if there
  // is more work to do
  gid += nxy;
  begin_z++;

  while (begin_z < end_z) {
    // Input data in front and back are shifted to discard the
    // oldest value and read one new value.
    for (auto iter = kHalfLength - 1; iter > 0; iter--) {
      back[iter] = back[iter - 1];
    }
    back[0] = front[0];

    for (auto iter = 0; iter < kHalfLength; iter++) {
      front[iter] = front[iter + 1];
    }

    // Only one new data-point read from global memory
    // in z-dimension (depth)
    front[kHalfLength] = prev[gid + kHalfLength * nxy];

    // Stencil code to update grid point at position given by global id (gid)
    float value = c[0] * front[0];
#pragma unroll(kHalfLength)
    for (auto iter = 1; iter <= kHalfLength; iter++) {
      value += c[iter] * (front[iter] + back[iter - 1] + prev[gid + iter] +
                          prev[gid - iter] + prev[gid + iter * nx] +
                          prev[gid - iter * nx]);
    }

    next[gid] = 2.0f * front[0] - next[gid] + value * vel[gid];

    gid += nxy;
    begin_z++;
  }
}

/*
 * Host-side SYCL Code
 *
 * Driver function for ISO3DFD SYCL code
 * Uses ptr_next and ptr_prev as ping-pong buffers to achieve
 * accelerated wave propogation
 *
 * This function uses SYCL buffers to facilitate host to device
 * buffer copies
 *
 */

bool Iso3dfdDevice(sycl::queue &q, float *ptr_next, float *ptr_prev,
                   float *ptr_vel, float *ptr_coeff, size_t n1, size_t n2,
                   size_t n3, size_t n1_block, size_t n2_block, size_t n3_block,
                   size_t end_z, unsigned int nIterations) {
  auto nx = n1;
  auto nxy = n1 * n2;

  auto bx = kHalfLength;
  auto by = kHalfLength;

  // Display information about the selected device
  PrintTargetInfo(q, n1_block, n2_block);

  auto grid_size = nxy * n3;

  {  // Begin buffer scope
    // Create buffers using DPC++ class buffer
    buffer b_ptr_next(ptr_next, range(grid_size));
    buffer b_ptr_prev(ptr_prev, range(grid_size));
    buffer b_ptr_vel(ptr_vel, range(grid_size));
    buffer b_ptr_coeff(ptr_coeff, range(kHalfLength + 1));

    // Iterate over time steps
    for (auto i = 0; i < nIterations; i += 1) {
      // Submit command group for execution
      q.submit([&](auto &h) {
        // Create accessors
        accessor next(b_ptr_next, h);
        accessor prev(b_ptr_prev, h);
        accessor vel(b_ptr_vel, h, read_only);
        accessor coeff(b_ptr_coeff, h, read_only);

        // Define local and global range

        // Define local ND range of work-items
        // Size of each DPC++ work-group selected here is a product of
        // n2_block and n1_block which can be controlled by the input
        // command line arguments
        auto local_nd_range = range(1, n2_block, n1_block);

        // Define global ND range of work-items
        // Size of total number of work-items is selected based on the
        // total grid size in first and second dimensions (XY-plane)
        //
        // Each of the work-item then works on computing
        // one or more grid points. This value can be controlled by the
        // input command line argument n3_block
        //
        // Effectively this implementation enables slicing of the full
        // grid into smaller grid slices which can be computed in parallel
        // to allow auto-scaling of the total number of work-items
        // spawned to achieve full occupancy for small or larger accelerator
        // devices
        auto global_nd_range =
            range((n3 - 2 * kHalfLength) / n3_block, (n2 - 2 * kHalfLength),
                  (n1 - 2 * kHalfLength));

#ifdef USE_SHARED
        // Using 3D-stencil kernel with Shared Local Memory (SLM)
        // optimizations (DPC++) to improve effective FLOPS to BYTES
        // ratio. By default, SLM code path is disabled in this
        // code sample.
        // SLM code path can be enabled by recompiling the DPC++ source
        // as follows:
        // cmake -DSHARED_KERNEL=1 ..
        // make -j`nproc`

        // Define a range for SLM Buffer
        // Padding can be used to avoid SLM bank conflicts
        // By default padding is disabled in the sample code
        auto local_range = range((n1_block + (2 * kHalfLength) + kPad) *
                                 (n2_block + (2 * kHalfLength)));

        //  Create an accessor for SLM buffer
        accessor<float, 1, access::mode::read_write, access::target::local> tab(
            local_range, h);

        // Send a DPC++ kernel (lambda) for parallel execution
        // The function that executes a single iteration is called
        // "Iso3dfdIterationSLM"
        // alternating the 'next' and 'prev' parameters which effectively
        // swaps their content at every iteration.
        if (i % 2 == 0)
          h.parallel_for(
              nd_range(global_nd_range, local_nd_range), [=](auto it) {
                Iso3dfdIterationSLM(it, next.get_pointer(), prev.get_pointer(),
                                    vel.get_pointer(), coeff.get_pointer(),
                                    tab.get_pointer(), nx, nxy, bx, by,
                                    n3_block, end_z);
              });
        else
          h.parallel_for(
              nd_range(global_nd_range, local_nd_range), [=](auto it) {
                Iso3dfdIterationSLM(it, prev.get_pointer(), next.get_pointer(),
                                    vel.get_pointer(), coeff.get_pointer(),
                                    tab.get_pointer(), nx, nxy, bx, by,
                                    n3_block, end_z);
              });

#else

        // Use Global Memory version of the 3D-Stencil kernel.
        // This code path is enabled by default

        // Send a DPC++ kernel (lambda) for parallel execution
        // The function that executes a single iteration is called
        // "Iso3dfdIterationGlobal"
        // alternating the 'next' and 'prev' parameters which effectively
        // swaps their content at every iteration.
        if (i % 2 == 0)
          h.parallel_for(
              nd_range(global_nd_range, local_nd_range), [=](auto it) {
                Iso3dfdIterationGlobal(it, next.get_pointer(),
                                       prev.get_pointer(), vel.get_pointer(),
                                       coeff.get_pointer(), nx, nxy, bx, by,
                                       n3_block, end_z);
              });
        else
          h.parallel_for(
              nd_range(global_nd_range, local_nd_range), [=](auto it) {
                Iso3dfdIterationGlobal(it, prev.get_pointer(),
                                       next.get_pointer(), vel.get_pointer(),
                                       coeff.get_pointer(), nx, nxy, bx, by,
                                       n3_block, end_z);
              });
#endif
      });
    }
  }  // end buffer scope
  return true;
}
