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

#include <mkl.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mkl_rng/distributions.hpp>
#include <mkl_rng_sycl.hpp>
#include <mkl_sycl_types.hpp>
#include <mkl_vml_sycl.hpp>
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

// Helper functions

// This function displays correct usage and parameters
void Usage(string program_name) {
  cout << " Incorrect number of parameters \n Usage: ";
  cout << program_name << " <Number of Iterations> <Seed for RNG> \n\n";
}

// This function prints a 1D vector as a matrix
template <typename T>
void PrintVectorAsMatrix(T* vector, size_t size_X, size_t size_Y) {
  cout << "\n";
  for (size_t j = 0; j < size_X; ++j) {
    for (size_t i = 0; i < size_Y; ++i) {
      cout << std::setw(3) << vector[j * size_Y + i] << " ";
    }
    cout << "\n";
  }
}

// This function prints a 2D matrix
template <typename T>
void PrintMatrix(T** matrix, size_t size_X, size_t size_Y) {
  cout << "\n";
  for (size_t i = 0; i < size_X; ++i) {
    for (size_t j = 0; j < size_Y; ++j) {
      cout << std::setw(3) << matrix[i][j] << " ";
    }
    cout << "\n";
  }
}

// This function prints a vector
template <typename T>
void PrintVector(T* vector, size_t n) {
  cout << "\n";
  for (size_t i = 0; i < n; ++i) {
    cout << vector[i] << " ";
  }
  cout << "\n";
}

// This function distributes simulation work across workers
void ParticleMotion(queue& q, size_t seed, float* particle_X, float* particle_Y,
                    size_t* grid, size_t grid_size, size_t n_particles,
                    unsigned int n_iterations, float radius) {
  auto device = q.get_device();
  auto maxBlockSize = device.get_info<info::device::max_work_group_size>();
  auto maxEUCount = device.get_info<info::device::max_compute_units>();

  // Total number of motion events
  const size_t n_moves = n_particles * n_iterations;

  cout << " Running on:: " << device.get_info<info::device::name>() << "\n";
  cout << " The Device Max Work Group Size is : " << maxBlockSize << "\n";
  cout << " The Device Max EUCount is : " << maxEUCount << "\n";
  cout << " The number of iterations is : " << n_iterations << "\n";
  cout << " The number of particles is : " << n_particles << "\n";

  // Declare vectors to store random values for X and Y directions
  float* random_X = new float[n_moves];
  float* random_Y = new float[n_moves];

  // Declare RNG object to compute vectors of random values
  // Basic random number generator object
  mkl::rng::philox4x32x10 engine(q, seed);
  // Distribution object
  mkl::rng::gaussian<float, mkl::rng::gaussian_method::box_muller2> distr(0.0, .03);

  // Begin scope for buffers
  {
    // Create buffers using DPC++ class buffer
    buffer b_random_X(random_X, range(n_moves));
    buffer b_random_Y(random_Y, range(n_moves));
    buffer b_particle_X(particle_X, range(n_particles));
    buffer b_particle_Y(particle_Y, range(n_particles));
    buffer b_grid(grid, range(grid_size * grid_size));

    // Compute vectors of random values for X and Y directions using RNG engine
    // declared above
    mkl::rng::generate(distr, engine, n_moves, b_random_X);
    mkl::rng::generate(distr, engine, n_moves, b_random_Y);

    // Submit command group for execution
    q.submit([&](handler& h) {
      auto a_particle_X = b_particle_X.get_access<access::mode::read_write>(h);
      auto a_particle_Y = b_particle_Y.get_access<access::mode::read_write>(h);
      auto a_random_X = b_random_X.get_access<access::mode::read>(h);
      auto a_random_Y = b_random_Y.get_access<access::mode::read>(h);
      // Atomic accessors: Use DPC++ atomic access mode
      auto a_grid = b_grid.get_access<access::mode::atomic>(h);

      // Send a DPC++ kernel (lambda) for parallel execution
      h.parallel_for(range(n_particles), [=](id<1> index) {
        int ii = index.get(0);
        float displacement_X = 0.0f;
        float displacement_Y = 0.0f;

        // Start iterations
        // Each iteration:
        //  1. Updates the position of all water molecules
        //  2. Checks if water molecule is inside a cell or not.
        //  3. Updates counter in cells array
        //
        for (size_t iter = 0; iter < n_iterations; iter++) {
          // Computes random displacement for each molecule
          // This example shows random distances between
          // -0.05 units and 0.05 units in both X and Y directions
          // Moves each water molecule by a random vector

          // Transform the random numbers into small displacements
          displacement_X = a_random_X[iter * n_particles + ii];
          displacement_Y = a_random_Y[iter * n_particles + ii];

          // Move particles using random displacements
          a_particle_X[ii] += displacement_X;
          a_particle_Y[ii] += displacement_Y;

          // Compute distances from particle position to grid point
          float dX = a_particle_X[ii] - sycl::trunc(a_particle_X[ii]);
          float dY = a_particle_Y[ii] - sycl::trunc(a_particle_Y[ii]);

          // Compute grid point indices
          int iX = sycl::floor(a_particle_X[ii]);
          int iY = sycl::floor(a_particle_Y[ii]);

          // Check if particle is still in computation grid
          if ((a_particle_X[ii] < grid_size) &&
              (a_particle_Y[ii] < grid_size) && (a_particle_X[ii] >= 0) &&
              (a_particle_Y[ii] >= 0)) {
            // Check if particle is (or remained) inside cell.
            // Increment cell counter in map array if so
            if ((dX * dX + dY * dY <= radius * radius)) {
              // Use DPC++ atomic_fetch_add to add 1 to accessor using atomic
              // mode
              atomic_fetch_add<size_t>(a_grid[iY * grid_size + iX], 1);
            }
          }
        }  // Next iteration
      });  // End parallel for
    });    // End queue submit.

  }  // End scope for buffers

  delete[] random_X;
  delete[] random_Y;
}  // End of function ParticleMotion()

int main(int argc, char* argv[]) {
  // Cell and Particle parameters
  const size_t grid_size = 21;    // Size of square grid
  const size_t n_particles = 20;  // Number of particles
  const float radius = 0.5f;      // Cell radius = 0.5*(grid spacing)

  // Default number of operations
  size_t n_iterations = 50;
  // Default seed for RNG
  size_t seed = 777;

  // Read command-line arguments
  try {
    n_iterations = stoi(argv[1]);
    seed = stoi(argv[2]);
  } catch (...) {
    Usage(argv[0]);
    return 1;
  }

  // Allocate arrays

  // Stores a grid of cells
  size_t* grid = new size_t[grid_size * grid_size];

  // Stores X and Y position of particles in the cell grid
  float* particle_X = new float[n_particles];
  float* particle_Y = new float[n_particles];

  // Initialize arrays
  for (size_t i = 0; i < n_particles; i++) {
    // Initial position of particles in cell grid
    particle_X[i] = 10.0f;
    particle_Y[i] = 10.0f;
  }

  for (size_t y = 0; y < grid_size; y++) {
    for (size_t x = 0; x < grid_size; x++) {
      grid[y * grid_size + x] = 0;
    }
  }

  // Create a device queue using default or host/gpu selectors
  default_selector device_selector;

  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler);

  // Start timers
  dpc_common::TimeInterval t_offload;

  // Call simulation function
  ParticleMotion(q, seed, particle_X, particle_Y, grid, grid_size, n_particles,
                 n_iterations, radius);

  q.wait_and_throw();

  // End timers
  auto time = t_offload.Elapsed();

  cout << "\n Offload time: " << time << " s\n\n";

  // Displays final grid only if grid small.
  if (grid_size <= 45) {
    cout << "\n ********************** OUTPUT GRID: \n ";
    PrintVectorAsMatrix<size_t>(grid, grid_size, grid_size);
  }

  // Cleanup
  delete[] grid;
  delete[] particle_X;
  delete[] particle_Y;

  return 0;
}
