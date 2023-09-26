//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// motionsim_kernel:
// Data Parallel C++ Language Basics Using a Monte Carlo
// Simulation (Device)
//

#include "motionsim.hpp"
namespace oneapi {}
using namespace oneapi;
using namespace sycl;
using namespace std;

template <typename AccRandom, typename AccGrid, bool HasFp64> class ParticleMotionKernel {
 public:
  ParticleMotionKernel(accessor<float> particle_X_a, accessor<float> particle_Y_a,
    AccRandom random_X_a, AccRandom random_Y_a, AccGrid grid_a, const size_t grid_size,
    const size_t n_particles, const size_t n_iterations, const size_t gs2, const float radius):
    particle_X_a(particle_X_a), particle_Y_a(particle_Y_a),
    random_X_a(random_X_a), random_Y_a(random_Y_a), grid_a(grid_a), grid_size(grid_size),
    n_particles(n_particles), n_iterations(n_iterations), gs2(gs2), radius(radius) {
     
    }
  void operator()(id<1> i) const {
    // Particle number (used for indexing)
    size_t p = i[0];
    // True when particle is found to be in a cell
    bool inside_cell = false;
    // Coordinates of the last known cell this particle resided in
    unsigned int prev_known_cell_coordinate_X;
    unsigned int prev_known_cell_coordinate_Y;

    // Motion simulation algorithm
    // --Start iterations--
    // Each iteration:
    //    1. Updates the position of all particles
    //    2. Checks if particle is inside a cell or not
    //    3. Updates counters in cells array (grid)
    //

    // Each particle performs this loop
    for (size_t iter = 0; iter < n_iterations; ++iter) {
      // Set the displacements to the random numbers
      float displacement_X = random_X_a[iter * n_particles + p];
      float displacement_Y = random_Y_a[iter * n_particles + p];
      // Displace particles
      particle_X_a[p] += displacement_X;
      particle_Y_a[p] += displacement_Y;
      // Compute distances from particle position to grid point i.e.,
      // the particle's distance from center of cell. Subtract the
      // integer value from floating point value to get just the
      // decimal portion. Use this value to later determine if the
      // particle is inside or outside of the cell
      float dX = sycl::abs(particle_X_a[p] - sycl::round(particle_X_a[p]));
      float dY = sycl::abs(particle_Y_a[p] - sycl::round(particle_Y_a[p]));
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
      int iX;
      int iY;
      if constexpr (HasFp64) {
        // Algorithm using double precision
        iX = sycl::floor(particle_X_a[p] + 0.5);
        iY = sycl::floor(particle_Y_a[p] + 0.5);
      } else {
        // Fall back code for single precision
        iX = sycl::floor(particle_X_a[p] + 0.5f);
        iY = sycl::floor(particle_Y_a[p] + 0.5f);
      }
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
                       --No action.                                      */

      // Atomic operations flags
      bool increment_C1 = false;
      bool increment_C2 = false;
      bool increment_C3 = false;
      bool decrement_C2_for_previous_cell = false;
      bool update_coordinates = false;

      // Check if particle's grid indices are still inside computation grid
      if ((iX < grid_size) && (iY < grid_size) && (iX >= 0) && (iY >= 0)) {
        // Compare the radius to particle's distance from center of cell
        if (radius >= sycl::sqrt(dX * dX + dY * dY)) {
          // Satisfies counter 1 requirement for cases 1, 3, 4
          increment_C1 = true;
          // Case 1
          if (!inside_cell) {
            increment_C2 = true;
            increment_C3 = true;
            inside_cell = true;
            update_coordinates = true;
          }
          // Case 3
          else if (prev_known_cell_coordinate_X != iX ||
                    prev_known_cell_coordinate_Y != iY) {
            increment_C2 = true;
            increment_C3 = true;
            update_coordinates = true;
            decrement_C2_for_previous_cell = true;
          }
          // Else: Case 4 --No action required. Counter 1 already updated

        }  // End inside cell if statement

        // Case 2a --Particle remained inside grid and moved outside cell
        else if (inside_cell) {
          inside_cell = false;
          decrement_C2_for_previous_cell = true;
        }
        // Else: Case 5a --Particle remained inside grid and outside cell
        // --No action required

      }  // End inside grid if statement

      // Case 2b --Particle moved outside grid and outside cell
      else if (inside_cell) {
        inside_cell = false;
        decrement_C2_for_previous_cell = true;
      }
      // Else: Case 5b --Particle remained outside of grid.
      // --No action required

      // Index variable for 3rd dimension of grid
      size_t layer;
      // Current and previous cell coordinates
      size_t curr_coordinates = iX + iY * grid_size;
      size_t prev_coordinates = prev_known_cell_coordinate_X +
                                prev_known_cell_coordinate_Y * grid_size;
      // gs2 variable (used below) equals grid_size * grid_size
      //

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = gs2;
      if (decrement_C2_for_previous_cell)
        atomic_fetch_sub<size_t>(grid_a[prev_coordinates + layer], 1);

      if (update_coordinates) {
        prev_known_cell_coordinate_X = iX;
        prev_known_cell_coordinate_Y = iY;
      }

      // Counter 1 layer of the grid (0 * grid_size * grid_size)
      layer = 0;
      if (increment_C1)
        atomic_fetch_add<size_t>(grid_a[curr_coordinates + layer], 1);

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = gs2;
      if (increment_C2)
      atomic_fetch_add<size_t>(grid_a[curr_coordinates + layer], 1);

      // Counter 3 layer of the grid (2 * grid_size * grid_size)
      layer = gs2 + gs2;
      if (increment_C3)
        atomic_fetch_add<size_t>(grid_a[curr_coordinates + layer], 1);

    }  // Next iteration
  }

  private:
    accessor<float> particle_X_a;
    accessor<float> particle_Y_a;
    AccRandom random_X_a;
    AccRandom random_Y_a;
    AccGrid grid_a;
    
    const size_t grid_size;
    const size_t n_particles;
    const size_t n_iterations;
    const size_t gs2;
    const float radius;
};

// This function distributes simulation work
void ParticleMotion(queue& q, const int seed, float* particle_X,
                    float* particle_Y, float* random_X, float* random_Y,
                    size_t* grid, const size_t grid_size, const size_t planes,
                    const size_t n_particles, const size_t n_iterations,
                    const float radius) {
  auto device = q.get_device();
  if (!device.has(aspect::fp64)) {
    std::cout << "Device " << device.get_info<info::device::name>() << " does not support double precision!"
      << " Single precision will be used instead." << std::endl;
  }
  auto maxBlockSize = device.get_info<info::device::max_work_group_size>();
  auto maxEUCount = device.get_info<info::device::max_compute_units>();
  // Total number of motion events
  const size_t n_moves = n_particles * n_iterations;
  // Grid size squared
  const size_t gs2 = grid_size * grid_size;

  cout << "Running on: " << device.get_info<info::device::name>() << "\n";
  cout << "Device Max Work Group Size: " << maxBlockSize << "\n";
  cout << "Device Max EUCount: " << maxEUCount << "\n";
  cout << "Number of iterations: " << n_iterations << "\n";
  cout << "Number of particles: " << n_particles << "\n";
  cout << "Size of the grid: " << grid_size << "\n";
  cout << "Random number seed: " << seed << "\n";

  // Declare basic random number generator (BRNG) for random vector
  mkl::rng::philox4x32x10 engine(q, seed);
  // Distribution object
  mkl::rng::gaussian<float, mkl::rng::gaussian_method::icdf> distr(alpha,
                                                                   sigma);
  // Begin buffer scope
  {
    // Create buffers using SYCL buffer class
    buffer random_X_buf(random_X, range(n_moves));
    buffer random_Y_buf(random_Y, range(n_moves));
    buffer particle_X_buf(particle_X, range(n_particles));
    buffer particle_Y_buf(particle_Y, range(n_particles));
    buffer grid_buf(grid, range(grid_size * grid_size * planes));

    // Compute random values using oneMKL RNG engine. Generates separate kernel
    mkl::rng::generate(distr, engine, n_moves, random_X_buf);
    mkl::rng::generate(distr, engine, n_moves, random_Y_buf);

    // Submit command group for execution
    // h is a handler type
    q.submit([&](auto& h) {
      // Declare accessors
      accessor particle_X_a(particle_X_buf, h);  // Read/write access
      accessor particle_Y_a(particle_Y_buf, h);
      accessor random_X_a(random_X_buf, h, read_only);
      accessor random_Y_a(random_Y_buf, h, read_only);
      // Use SYCL atomic access mode to create atomic accessors
      accessor grid_a = grid_buf.get_access<access::mode::atomic>(h);

      if (device.has(aspect::fp64)) {
        h.parallel_for(range (n_particles ),
                        ParticleMotionKernel<decltype(random_X_a), decltype(grid_a), true>(particle_X_a, particle_Y_a,
                          random_X_a, random_Y_a, grid_a, grid_size, n_particles, n_iterations, gs2, radius));
      } else {
        h.parallel_for(range (n_particles),
                        ParticleMotionKernel<decltype(random_X_a), decltype(grid_a), false>(particle_X_a, particle_Y_a,
                        random_X_a, random_Y_a, grid_a, grid_size, n_particles, n_iterations, gs2, radius));
      }     // End parallel for
    });     // End queue submit. End accessor scope
  }         // End buffer scope
}  // End of function ParticleMotion()
