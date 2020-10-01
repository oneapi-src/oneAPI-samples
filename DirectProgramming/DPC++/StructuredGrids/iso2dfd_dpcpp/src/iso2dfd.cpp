//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// ISO2DFD: Intel® oneAPI DPC++ Language Basics Using 2D-Finite-Difference-Wave
// Propagation
//
// ISO2DFD is a finite difference stencil kernel for solving the 2D acoustic
// isotropic wave equation. Kernels in this sample are implemented as 2nd order
// in space, 2nd order in time scheme without boundary conditions. Using Data
// Parallel C++, the sample will explicitly run on the GPU as well as CPU to
// calculate a result.  If successful, the output will include GPU device name.
//
// A complete online tutorial for this code sample can be found at :
// https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide 
// and search based on relevant terms noted in the comments.
//
// DPC++ material used in this code sample:
//
// Basic structures of DPC++:
//   DPC++ Queues (including device selectors and exception handlers)
//   DPC++ Buffers and accessors (communicate data between the host and the device)
//   DPC++ Kernels (including parallel_for function and range<2> objects)
//

#include <fstream>
#include <iostream>
#include <CL/sycl.hpp>
#include <cmath>
#include <cstring>
#include <stdio.h>

#include "dpc_common.hpp"

using namespace cl::sycl;
using namespace std;

/*
 * Parameters to define coefficients
 * half_length: Radius of the stencil
 * Sample source code is tested for half_length=1 resulting in
 * 2nd order Stencil finite difference kernel
 */

constexpr float DT = 0.002f;
constexpr float DXY = 20.0f;
constexpr unsigned int half_length = 1;

/*
 * Host-Code
 * Utility function to display input arguments
 */
void Usage(const string &program_name) {
  cout << " Incorrect parameters\n";
  cout << " Usage: ";
  cout << program_name << " n1 n2 Iterations\n\n";
  cout << " n1 n2      : Grid sizes for the stencil\n";
  cout << " Iterations : No. of timesteps.\n";
}

/*
 * Host-Code
 * Function used for initialization
 */
void Initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, size_t n_rows,
                size_t n_cols) {
  cout << "Initializing ...\n";

  // Define source wavelet
  float wavelet[12] = {0.016387336, -0.041464937, -0.067372555, 0.386110067,
                       0.812723635, 0.416998396,  0.076488599,  -0.059434419,
                       0.023680172, 0.005611435,  0.001823209,  -0.000720549};

  // Initialize arrays
  for (size_t i = 0; i < n_rows; i++) {
    size_t offset = i * n_cols;

    for (int k = 0; k < n_cols; k++) {
      ptr_prev[offset + k] = 0.0f;
      ptr_next[offset + k] = 0.0f;
      // pre-compute squared value of sample wave velocity v*v (v = 1500 m/s)
      ptr_vel[offset + k] = (1500.0f * 1500.0f);
    }
  }
  // Add a source to initial wavefield as an initial condition
  for (int s = 11; s >= 0; s--) {
    for (int i = n_rows / 2 - s; i < n_rows / 2 + s; i++) {
      size_t offset = i * n_cols;
      for (int k = n_cols / 2 - s; k < n_cols / 2 + s; k++) {
        ptr_prev[offset + k] = wavelet[s];
      }
    }
  }
}

/*
 * Host-Code
 * Utility function to print device info
 */
void PrintTargetInfo(queue& q) {
  auto device = q.get_device();
  auto max_block_size =
      device.get_info<info::device::max_work_group_size>();

  auto max_EU_count =
      device.get_info<info::device::max_compute_units>();

  cout<< " Running on " << device.get_info<info::device::name>()<<"\n";
  cout<< " The Device Max Work Group Size is : "<< max_block_size<<"\n";
  cout<< " The Device Max EUCount is : " << max_EU_count<<"\n";
}

/*
 * Host-Code
 * Utility function to calculate L2-norm between resulting buffer and reference
 * buffer
 */
bool WithinEpsilon(float* output, float* reference, const size_t dim_x,
                    const size_t dim_y, const unsigned int radius,
                    const float delta = 0.01f) {
  ofstream err_file;
  err_file.open("error_diff.txt");

  bool error = false;
  double norm2 = 0;

  for (size_t iy = 0; iy < dim_y; iy++) {
    for (size_t ix = 0; ix < dim_x; ix++) {
      if (ix >= radius && ix < (dim_x - radius) && iy >= radius &&
          iy < (dim_y - radius)) {
        float difference = fabsf(*reference - *output);
        norm2 += difference * difference;
        if (difference > delta) {
          error = true;
          err_file<<" ERROR: "<<ix<<", "<<iy<<"   "<<*output<<"   instead of "<<
                   *reference<<"  (|e|="<<difference<<")\n";
        }
      }

      ++output;
      ++reference;
    }
  }

  err_file.close();
  norm2 = sqrt(norm2);
  if (error) printf("error (Euclidean norm): %.9e\n", norm2);
  return error;
}

/*
 * Host-Code
 * CPU implementation for wavefield modeling
 * Updates wavefield for the number of iterations given in nIteratons parameter
 */
void Iso2dfdIterationCpu(float* next, float* prev, float* vel,
                            const float dtDIVdxy, int n_rows, int n_cols,
                            int n_iterations) {
  float* swap;
  float value = 0.0;
  int   gid = 0;
  for (unsigned int k = 0; k < n_iterations; k += 1) {
    for (unsigned int i = 1; i < n_rows - half_length; i += 1) {
      for (unsigned int j = 1; j < n_cols - half_length; j += 1) {
        value = 0.0;

        // Stencil code to update grid
        gid = j + (i * n_cols);
        value = 0.0;
        value += prev[gid + 1] - 2.0 * prev[gid] + prev[gid - 1];
        value += prev[gid + n_cols] - 2.0 * prev[gid] + prev[gid - n_cols];
        value *= dtDIVdxy * vel[gid];
        next[gid] = 2.0f * prev[gid] - next[gid] + value;
      }
    }

    // Swap arrays
    swap = next;
    next = prev;
    prev = swap;
  }
}

/*
 * Device-Code - GPU
 * SYCL implementation for single iteration of iso2dfd kernel
 *
 * Range kernel is used to spawn work-items in x, y dimension
 *
 */
void Iso2dfdIterationGlobal(id<2> it, float* next, float* prev,
                               float* vel, const float dtDIVdxy, int n_rows,
                               int n_cols) {
  float value = 0.0;

  // Compute global id
  // We can use the get.global.id() function of the item variable
  //   to compute global id. The 2D array is laid out in memory in row major
  //   order.
  size_t gid_row = it.get(0);
  size_t gid_col = it.get(1);
  size_t gid = (gid_row)*n_cols + gid_col;

  // Computation to solve wave equation in 2D
  // First check if gid is inside the effective grid (not in halo)
  if ((gid_col >= half_length && gid_col < n_cols - half_length) &&
      (gid_row >= half_length && gid_row < n_rows - half_length)) {
    // Stencil code to update grid point at position given by global id (gid)
    // New time step for grid point is computed based on the values of the
    //    the immediate neighbors in both the horizontal and vertical
    //    directions, as well as the value of grid point at a previous time step
    value = 0.0;
    value += prev[gid + 1] - 2.0 * prev[gid] + prev[gid - 1];
    value += prev[gid + n_cols] - 2.0 * prev[gid] + prev[gid - n_cols];
    value *= dtDIVdxy * vel[gid];
    next[gid] = 2.0f * prev[gid] - next[gid] + value;
  }
}

int main(int argc, char* argv[]) {
  // Arrays used to update the wavefield
  float* prev_base;
  float* next_base;
  float* next_cpu;
  // Array to store wave velocity
  float* vel_base;

  bool error = false;

  size_t n_rows, n_cols;
  unsigned int n_iterations;

  // Read parameters
  try {
    n_rows = stoi(argv[1]);
    n_cols = stoi(argv[2]);
    n_iterations = stoi(argv[3]);
  }

  catch (...) {
    Usage(argv[0]);
    return 1;
  }

  // Compute the total size of grid
  size_t n_size = n_rows * n_cols;

  // Allocate arrays to hold wavefield and velocity
  prev_base = new float[n_size];
  next_base = new float[n_size];
  next_cpu = new float[n_size];
  vel_base = new float[n_size];

  // Compute constant value (delta t)^2 (delta x)^2. To be used in wavefield
  // update
  float dtDIVdxy = (DT * DT) / (DXY * DXY);

  // Initialize arrays and introduce initial conditions (source)
  Initialize(prev_base, next_base, vel_base, n_rows, n_cols);

  cout << "Grid Sizes: " << n_rows << " " << n_cols << "\n";
  cout << "Iterations: " << n_iterations << "\n\n";

  // Define device selector as 'default'
  default_selector device_selector;

  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler);

  cout << "Computing wavefield in device ..\n";
  // Display info about device
  PrintTargetInfo(q);

  // Start timer
  dpc_common::TimeInterval t_offload;

  {  // Begin buffer scope
    // Create buffers using DPC++ class buffer
    buffer next_buf(next_base, range(n_size));
    buffer prev_buf(prev_base, range(n_size));
    buffer vel_buf(vel_base, range(n_size));

    // Iterate over time steps
    for (unsigned int k = 0; k < n_iterations; k += 1) {
      // Submit command group for execution
      q.submit([&](auto &h) {
        // Create accessors
        accessor next_a(next_buf, h);
        accessor prev_a(prev_buf, h);
        accessor vel_a(vel_buf, h, read_only);

        // Define local and global range
        auto global_range = range<2>(n_rows, n_cols);

        // Send a DPC++ kernel (lambda) for parallel execution
        // The function that executes a single iteration is called
        // "iso_2dfd_iteration_global"
        //    alternating the 'next' and 'prev' parameters which effectively
        //    swaps their content at every iteration.
        if (k % 2 == 0)
          h.parallel_for(global_range, [=](auto it) {
                Iso2dfdIterationGlobal(it, next_a.get_pointer(),
                                          prev_a.get_pointer(), vel_a.get_pointer(),
                                          dtDIVdxy, n_rows, n_cols);
              });
        else
          h.parallel_for(global_range, [=](auto it) {
                Iso2dfdIterationGlobal(it, prev_a.get_pointer(),
                                          next_a.get_pointer(), vel_a.get_pointer(),
                                          dtDIVdxy, n_rows, n_cols);
              });
      });

    }  // end for

  }  // buffer scope

  // Wait for commands to complete. Enforce synchronization on the command queue
  q.wait_and_throw();

  // Compute and display time used by device
  auto time = t_offload.Elapsed();

  cout << "Offload time: " << time << " s\n\n";

  // Output final wavefield (computed by device) to binary file
  ofstream out_file;
  out_file.open("wavefield_snapshot.bin", ios::out | ios::binary);
  out_file.write(reinterpret_cast<char*>(next_base), n_size * sizeof(float));
  out_file.close();

  // Compute wavefield on CPU (for validation)
  
  cout << "Computing wavefield in CPU ..\n";
  // Re-initialize arrays
  Initialize(prev_base, next_cpu, vel_base, n_rows, n_cols);

  // Compute wavefield on CPU
  // Start timer for CPU
  dpc_common::TimeInterval t_cpu;

  Iso2dfdIterationCpu(next_cpu, prev_base, vel_base, dtDIVdxy, n_rows, n_cols,
                         n_iterations);

  // Compute and display time used by CPU
  time = t_cpu.Elapsed();

  cout << "CPU time: " << time << " s\n\n";

  // Compute error (difference between final wavefields computed in device and
  // CPU)
  error = WithinEpsilon(next_base, next_cpu, n_rows, n_cols, half_length, 0.1f);

  // If error greater than threshold (last parameter in error function), report
  if (error)
    cout << "Final wavefields from device and CPU are different: Error\n";
  else
    cout << "Final wavefields from device and CPU are equivalent: Success\n";

  // Output final wavefield (computed by CPU) to binary file
  out_file.open("wavefield_snapshot_cpu.bin", ios::out | ios::binary);
  out_file.write(reinterpret_cast<char*>(next_cpu), n_size * sizeof(float));
  out_file.close();

  cout << "Final wavefields (from device and CPU) written to disk\n";
  cout << "Finished.\n";

  // Cleanup
  delete[] prev_base;
  delete[] next_base;
  delete[] vel_base;

  return error ? 1 : 0;
}
