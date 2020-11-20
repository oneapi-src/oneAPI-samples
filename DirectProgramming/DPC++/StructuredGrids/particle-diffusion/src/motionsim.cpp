//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// motionsim: Intel® oneAPI DPC++ Language Basics Using a Monte Carlo
// Simulation
//
// This code sample will implement a simple example of a Monte Carlo
// simulation of the diffusion of water molecules in tissue.
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide
// and search based on relevant terms noted in the comments.
//
// DPC++ material used in this code sample:
//
// Basic structures of DPC++:
//   DPC++ Queues (including device selectors and exception handlers)
//   DPC++ Buffers and accessors (communicate data between the host and the
//   device) DPC++ Kernels (including parallel_for function and range<2>
//   objects) API-based programming: Use oneMKL to generate random numbers
//   (DPC++) DPC++ atomic operations for synchronization
//

#include "motionsim.hpp"
namespace oneapi {}
using namespace oneapi;
using namespace sycl;
using namespace std;

// This function distributes simulation work
void CPUParticleMotion(const int seed, float* particle_X, float* particle_Y,
                       float* random_X, float* random_Y, size_t* grid,
                       const size_t grid_size, const size_t planes,
                       const size_t n_particles, unsigned int n_iterations,
                       const float radius) {
  // Grid size squared
  const size_t gs2 = grid_size * grid_size;

  cout << "Running on: CPU\n";
  cout << "Number of iterations: " << n_iterations << "\n";
  cout << "Number of particles: " << n_particles << "\n";
  cout << "Size of the grid: " << grid_size << "\n";
  cout << "Random number seed: " << seed << "\n";

  // Array of flags for each particle.
  // Each array initialized to false/zero with new array zero initializer.
  // True when particle is found to be in a cell
  bool* inside_cell = new bool[n_particles]();
  // Operations flags
  bool* increment_C1 = new bool[n_particles]();
  bool* increment_C2 = new bool[n_particles]();
  bool* increment_C3 = new bool[n_particles]();
  bool* decrement_C2_for_previous_cell = new bool[n_particles]();
  bool* update_coordinates = new bool[n_particles]();
  // Coordinates of the last known cell this particle resided in.
  // Initialized in motion simulation algorithm below and presumably not used
  // before initialization. Thus, avoid spending computation time initializing
  unsigned int* prev_known_cell_coordinate_X = new unsigned int[n_particles];
  unsigned int* prev_known_cell_coordinate_Y = new unsigned int[n_particles];

  // Motion simulation algorithm
  // --Start iterations--
  // Each iteration:
  //    1. Updates the position of all particles
  //    2. Checks if particle is inside a cell or not
  //    3. Updates counters in cells array (grid)
  //

  // All n_particles particles need to each be displaced once per iteration
  // to match device's algorithm
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    for (size_t p = 0; p < n_particles; ++p) {
      // Set the displacements to the random numbers
      float displacement_X = random_X[iter * n_particles + p];
      float displacement_Y = random_Y[iter * n_particles + p];
      // Displace particles
      particle_X[p] += displacement_X;
      particle_Y[p] += displacement_Y;
      // Compute distances from particle position to grid point i.e.,
      // the particle's distance from center of cell. Subtract the
      // integer value from floating point value to get just the
      // decimal portion. Use this value to later determine if the
      // particle is inside or outside of the cell
      float dX = sycl::abs(particle_X[p] - sycl::round(particle_X[p]));
      float dY = sycl::abs(particle_Y[p] - sycl::round(particle_Y[p]));
      /* Grid point indices closest the particle, defined by the following:
      ------------------------------------------------------------------
      |               Condition               |         Result         |
      |---------------------------------------|------------------------|
      |particle_X + 0.5 >= ceiling(particle_X)|iX = ceiling(particle_X)|
      |---------------------------------------|------------------------|
      |particle_Y + 0.5 >= ceiling(particle_Y)|iY = ceiling(particle_Y)|
      |---------------------------------------|------------------------|
      |particle_X + 0.5 < ceiling(particle_X) |iX = floor(particle_X)  |
      |---------------------------------------|------------------------|
      |particle_Y + 0.5 < ceiling(particle_Y) |iY = floor(particle_Y)  |
      ------------------------------------------------------------------  */
      int iX = sycl::floor(particle_X[p] + 0.5);
      int iY = sycl::floor(particle_Y[p] + 0.5);

      /* There are 5 cases when considering particle movement about the
         grid.

         All 5 cases are distinct from one another; i.e., any particle's
         motion falls under one and only one of the following cases:

           Case 1: Particle moves from outside cell to inside cell
                   --Increment counters 1-3
                   --Turn on inside_cell flag
                   --Store the coordinates of the
                     particle's new cell location

           Case 2: Particle moves from inside cell to outside
                   cell (and possibly outside of the grid)
                   --Decrement counter 2 for old cell
                   --Turn off inside_cell flag

           Case 3: Particle moves from inside one cell to inside
                   another cell
                   --Decrement counter 2 for old cell
                   --Increment counters 1-3 for new cell
                   --Store the coordinates of the particle's new cell
                     location

           Case 4: Particle moves and remains inside original
                   cell (does not leave cell)
                   --Increment counter 1

           Case 5: Particle moves and remains outside of cell
                   --No action.                                       */

      // Check if particle is still in computation grid
      if ((iX < grid_size) && (iY < grid_size) && (iX >= 0) && (iY >= 0)) {
        // Compare the radius to particle's distance from center of cell
        if (radius >= sycl::sqrt(dX * dX + dY * dY)) {
          // Satisfies counter 1 requirement for cases 1, 3, 4
          increment_C1[p] = true;
          // Case 1
          if (!inside_cell[p]) {
            increment_C2[p] = true;
            increment_C3[p] = true;
            inside_cell[p] = true;
            update_coordinates[p] = true;
          }
          // Case 3
          else if (prev_known_cell_coordinate_X[p] != iX ||
                   prev_known_cell_coordinate_Y[p] != iY) {
            increment_C2[p] = true;
            increment_C3[p] = true;
            update_coordinates[p] = true;
            decrement_C2_for_previous_cell[p] = true;
          }
          // Else: Case 4 --No action required. Counter 1 already updated

        }  // End inside cell if statement

        // Case 2a --Particle remained inside grid and moved outside cell
        else if (inside_cell[p]) {
          inside_cell[p] = false;
          decrement_C2_for_previous_cell[p] = true;
        }
        // Else: Case 5a --Particle remained inside grid and outside cell
        // --No action required

      }  // End inside grid if statement

      // Case 2b --Particle moved outside grid and outside cell
      else if (inside_cell[p]) {
        inside_cell[p] = false;
        decrement_C2_for_previous_cell[p] = true;
      }
      // Else: Case 5b --Particle remained outside of grid
      // --No action required

      // Index variable for 3rd dimension of grid
      size_t layer;
      // Current and previous cell coordinates
      size_t curr_coordinates = iX + iY * grid_size;
      size_t prev_coordinates = prev_known_cell_coordinate_X[p] +
                                prev_known_cell_coordinate_Y[p] * grid_size;
      // gs2 variable (used below) equals grid_size * grid_size
      //

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = gs2;
      if (decrement_C2_for_previous_cell[p]) --(grid[prev_coordinates + layer]);

      if (update_coordinates[p]) {
        prev_known_cell_coordinate_X[p] = iX;
        prev_known_cell_coordinate_Y[p] = iY;
      }

      // Counter 1 layer of the grid (0 * grid_size * grid_size)
      layer = 0;
      if (increment_C1[p]) ++(grid[curr_coordinates + layer]);

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = gs2;
      if (increment_C2[p]) ++(grid[curr_coordinates + layer]);

      // Counter 3 layer of the grid (2 * grid_size * grid_size)
      layer = gs2 + gs2;
      if (increment_C3[p]) ++(grid[curr_coordinates + layer]);

      increment_C1[p] = false;
      increment_C2[p] = false;
      increment_C3[p] = false;
      decrement_C2_for_previous_cell[p] = false;
      update_coordinates[p] = false;

    }  // Next iteration inner for loop
  }    // Next iteration outer for loop
  delete[] inside_cell;
  delete[] increment_C1;
  delete[] increment_C2;
  delete[] increment_C3;
  delete[] decrement_C2_for_previous_cell;
  delete[] update_coordinates;
  delete[] prev_known_cell_coordinate_X;
  delete[] prev_known_cell_coordinate_Y;
}  // End of function CPUParticleMotion()

// Main Function
int main(int argc, char* argv[]) {
  // Set command line arguments to their default values
  size_t n_iterations = 10000;
  size_t n_particles = 256;
  size_t grid_size = 22;
  int seed = 777;
  unsigned int cpu_flag = 0;
  unsigned int grid_output_flag = 1;

  cout << "\n";
  if (argc == 1)
    cout << "**Running with default parameters**\n\n";
  else {
    int rc = 0;
// Detect OS type and read in command line arguments
#if !WINDOWS
    rc = ParseArgs(argc, argv, &n_iterations, &n_particles, &grid_size, &seed,
                   &cpu_flag, &grid_output_flag);
#elif WINDOWS  // WINDOWS
    rc = ParseArgsWindows(argc, argv, &n_iterations, &n_particles, &grid_size,
                          &seed, &cpu_flag, &grid_output_flag);
#else          // WINDOWS
    cout << "Error. Failed to detect operating system. Exiting.\n";
    return 1;
#endif         // WINDOWS
    if (rc != 0) return 1;
  }  // End else

  // Allocate and initialize arrays
  //

  // Stores X and Y position of particles in the cell grid
  float* particle_X = new float[n_particles];
  float* particle_Y = new float[n_particles];
  // Total number of motion events
  const size_t n_moves = n_particles * n_iterations;
  // Declare vectors to store random values for X and Y directions
  float* random_X = new float[n_moves];
  float* random_Y = new float[n_moves];
  // Grid center
  const float center = grid_size / 2;
  // Initialize the particle starting positions to the grid center
  for (size_t i = 0; i < n_particles; ++i) {
    particle_X[i] = center;
    particle_Y[i] = center;
  }

  /*  Each of the folowing counters represent a separate plane in the grid
  variable (described below).
  Each plane is of size: grid_size * grid_size. There are 3 planes.

  _________________________________________________________________________
  |COUNTER                                                           PLANE|
  |-----------------------------------------------------------------------|
  |counter 1: Total particle accumulation counts for each cell            |
  |           throughout the entire simulation. Includes             z = 0|
  |           particles that remained in, as well as particles            |
  |           that have returned to each cell.                            |
  |                                                                       |
  |counter 2: Current total number of particles in each cell.        z = 1|
  |           Does not increase if a particle remained in a cell.         |
  |           An instantaneous snapshot of the grid of cells.             |
  |                                                                       |
  |counter 3: Total number of particle entries into each cell.       z = 2|
  |_______________________________________________________________________|

  The 3D matrix is implemented as a 1D matrix to improve efficiency.

  For any given index j with coordinates (x, y, z) in a N x M x D (Row x
  Column x Depth) 3D matrix, the same index j in an equivalently sized 1D
  matrix is given by the following formula:

  j = y + M * x + M * M * z                                              */

  // Size of 3rd dimension of 3D grid (3D matrix). 3 counters => 3 planes
  const size_t planes = 3;
  // Stores a grid of cells, initialized to zero.
  size_t* grid = new size_t[grid_size * grid_size * planes]();
  // Cell radius = 0.5*(grid spacing)
  const float radius = 0.5f;

  // Create a device queue using default or host/device selectors
  default_selector device_selector;
  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler);

  // Start timers
  dpc_common::TimeInterval t_offload;
  // Call device simulation function
  ParticleMotion(q, seed, particle_X, particle_Y, random_X, random_Y, grid,
                 grid_size, planes, n_particles, n_iterations, radius);
  q.wait_and_throw();
  auto device_time = t_offload.Elapsed();
  // End timers

  cout << "\nDevice Offload time: " << device_time << " s\n\n";

  size_t* grid_cpu;
  // If user wants to perform cpu computation, for comparison with device
  // Off by default
  if (cpu_flag == 1) {
    // Re-initialize arrays
    for (size_t i = 0; i < n_particles; ++i) {
      particle_X[i] = center;
      particle_Y[i] = center;
    }

    // Use Math Kernel Library (MKL) VSL Gaussian function for RNG with
    // mean of alpha and standard deviation of sigma
    //

    VSLStreamStatePtr stream;
    int vsl_retv = vslNewStream(&stream, VSL_BRNG_PHILOX4X32X10, seed);
    CheckVslError(vsl_retv);

    vsl_retv = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n_moves,
                             random_X, alpha, sigma);
    CheckVslError(vsl_retv);

    vsl_retv = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n_moves,
                             random_Y, alpha, sigma);
    CheckVslError(vsl_retv);

    grid_cpu = new size_t[grid_size * grid_size * planes]();

    // Start timers
    dpc_common::TimeInterval t_offload_cpu;
    // Call CPU simulation function
    CPUParticleMotion(seed, particle_X, particle_Y, random_X, random_Y,
                      grid_cpu, grid_size, planes, n_particles, n_iterations,
                      radius);
    auto cpu_time = t_offload_cpu.Elapsed();
    // End timers

    cout << "\nCPU Offload time: " << cpu_time << " s\n\n";
  }
  if (grid_output_flag)
    PrintGrids(grid, grid_cpu, grid_size, cpu_flag, grid_output_flag);
  if (cpu_flag)
    PrintValidationResults(grid, grid_cpu, grid_size, planes, cpu_flag,
                           grid_output_flag);
  else
    cout << "Success.\n";

  // Cleanup
  if (cpu_flag) delete[] grid_cpu;
  delete[] grid;
  delete[] random_X;
  delete[] random_Y;
  delete[] particle_X;
  delete[] particle_Y;
  return 0;
}  // End of function main()
