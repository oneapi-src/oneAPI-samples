//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// motionsim_kernel:
// Intel® oneAPI DPC++ Language Basics Using a Monte Carlo
// Simulation (Device)
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide
// and search based on relevant terms noted in the comments.
//
// Basic structures of DPC++ used in this code sample:
//
//   -Queues: device selectors and exception handlers
//   -Buffers and accessors: communication between host and device
//   -Kernels: parallel_for function and range<n> objects
//   -API-based programming: Use of oneMKL to generate random numbers
//   -Atomic operations: synchronization in device
//

// This function distributes simulation work across workers
void ParticleMotion(queue& q, const size_t seed, float* particle_X,
                    float* particle_Y, float* random_X, float* random_Y,
                    size_t* grid, const size_t grid_size, const size_t planes,
                    const size_t n_particles, const unsigned int n_iterations,
                    const float radius) {
  auto device = q.get_device();
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
  oneapi::mkl::rng::philox4x32x10 engine(q, seed);
  // Distribution object
  oneapi::mkl::rng::gaussian<float, oneapi::mkl::rng::gaussian_method::icdf>
      distr(ALPHA, SIGMA);
  // Begin buffer scope
  {
    // Create buffers using DPC++ buffer class
    buffer random_X_buf(random_X, range(n_moves));
    buffer random_Y_buf(random_Y, range(n_moves));
    buffer particle_X_buf(particle_X, range(n_particles));
    buffer particle_Y_buf(particle_Y, range(n_particles));
    buffer grid_buf(grid, range(grid_size * grid_size * planes));

    // Compute random values using oneMKL RNG engine. Generates separate kernel
    oneapi::mkl::rng::generate(distr, engine, n_moves, random_X_buf);
    oneapi::mkl::rng::generate(distr, engine, n_moves, random_Y_buf);

    // Submit command group for execution
    // h is a handler type
    q.submit([&](auto& h) {
      // Declare accessors
      accessor particle_X_a =
          particle_X_buf.get_access<access::mode::read_write>(h);
      accessor particle_Y_a =
          particle_Y_buf.get_access<access::mode::read_write>(h);
      accessor random_X_a = random_X_buf.get_access<access::mode::read>(h);
      accessor random_Y_a = random_Y_buf.get_access<access::mode::read>(h);
      // Use DPC++ atomic access mode to create atomic accessors
      accessor grid_a = grid_buf.get_access<access::mode::atomic>(h);

      // Send a DPC++ kernel (lambda) for parallel execution.
      h.parallel_for(range(n_particles), [=](auto item) {
        // Particle number for indexing
        size_t p = item.get_id(0);

        // True if particle is inside the cell radius
        bool within_radius;
        float displacement_X = 0.0f;
        float displacement_Y = 0.0f;
        // Flag is true when particle is in cell
        bool inside_cell = false;
        // Atomic operations flags
        bool increment_C1 = false, increment_C2 = false, increment_C3 = false,
             decrement_C2 = false, update_coordinates = false;

        // Current coordinates of the particle
        int iX, iY;
        // Coordinates of the last known cell this particle resided in
        unsigned int prev_known_cell_coordinate_X;
        unsigned int prev_known_cell_coordinate_Y;
        // Index variable for 3rd dimension of grid
        size_t layer = 0;
        // For matrix indexing
        size_t curr_cell_coordinates, prev_cell_coordinates;

        // --Start iterations--
        // Each iteration:
        //    1. Updates the position of all particles
        //    2. Checks if particle is inside a cell or not.
        //    3. Updates counters in cells array (grid)

        // Each particle performs this loop
        for (size_t iter = 0; iter < n_iterations; ++iter) {
          // Compute random displacement for each particle.
          // This example uses random displacements between
          // -0.05 units and 0.05 units in X, Y directions.
          // Moves each particle by a random vector.

          // Transform the random numbers into small displacements
          displacement_X = random_X_a[iter * n_particles + p];
          displacement_Y = random_Y_a[iter * n_particles + p];

          // Displace particles
          particle_X_a[p] += displacement_X;
          particle_Y_a[p] += displacement_Y;

          // Compute distances from particle position to grid point.
          // I.e., the particle's distance from center of cell. Subtract the
          // integer value from floating point value to get just
          // the decimal portion. Use this value to later determine if the
          // particle is in the cell or outside the cell.
          float dX = particle_X_a[p] - sycl::trunc(particle_X_a[p]);
          float dY = particle_Y_a[p] - sycl::trunc(particle_Y_a[p]);

          // Compute grid point indices
          iX = sycl::floor(particle_X_a[p]);
          iY = sycl::floor(particle_Y_a[p]);

          /* There are 5 cases when considering particle movement about the
             grid.

             All 5 cases are distinct from one another; i.e., any
             particle's motion falls under one and only one of the following
             cases:

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
                       --No action.
          */

          // Check if particle is still in computation grid
          if ((particle_X_a[p] < grid_size) && (particle_Y_a[p] < grid_size) &&
              (particle_X_a[p] >= 0) && (particle_Y_a[p] >= 0)) {
            // Compare the radius to the magnitude of the particle's distance
            // from the center of the cell
            within_radius =
                radius >= sycl::sqrt(dX * dX + dY * dY) ? true : false;

            // Check if particle is in cell
            // Either case 1, 3, or 4
            if (within_radius) {
              // Completes this step for cases 1, 3, 4
              increment_C1 = true;
              // Case 1
              if (!inside_cell) {
                increment_C2 = true;
                increment_C3 = true;
                inside_cell = true;
                update_coordinates = true;
              }

              // Either case 3 or 4
              // Note: Regardless of if this is case 3 or 4, counter 1 has
              // already been incremented above.
              else if (inside_cell) {
                // Case 3
                if (prev_known_cell_coordinate_X != iX ||
                    prev_known_cell_coordinate_Y != iY) {
                  decrement_C2 = true;
                  increment_C2 = true;
                  increment_C3 = true;
                  update_coordinates = true;
                }
                // Case 4
              }
            }  // End inside cell if statement

            // Either case 2 or 5
            else {
              // Case 2
              if (inside_cell) {
                inside_cell = false;
                decrement_C2 = true;
              }
              // Case 5
            }
          }  // End inside grid if statement

          // Either case 2 or 5
          else {
            // Case 2
            if (inside_cell) {
              inside_cell = false;
              decrement_C2 = true;
            }
            // Case 5
          }

          if (update_coordinates) {
            prev_known_cell_coordinate_X = iX;
            prev_known_cell_coordinate_Y = iY;
          }

          // Matrix index calculations for the current and previous cell
          curr_cell_coordinates = iX + iY * grid_size;
          prev_cell_coordinates = prev_known_cell_coordinate_X +
                                  prev_known_cell_coordinate_Y * grid_size;

          // Counter 1 layer of the grid (0 * grid_size * grid_size)
          layer = 0;
          if (increment_C1)
            atomic_fetch_add<size_t>(grid_a[curr_cell_coordinates + layer], 1);

          // Counter 2 layer of the grid (1 * grid_size * grid_size)
          layer = gs2;
          if (increment_C2)
            atomic_fetch_add<size_t>(grid_a[curr_cell_coordinates + layer], 1);

          // Counter 3 layer of the grid (2 * grid_size * grid_size)
          layer = gs2 + gs2;
          if (increment_C3)
            atomic_fetch_add<size_t>(grid_a[curr_cell_coordinates + layer], 1);

          // Counter 2 layer of the grid (1 * grid_size * grid_size)
          layer = gs2;
          if (decrement_C2)
            atomic_fetch_sub<size_t>(grid_a[prev_cell_coordinates + layer], 1);

          increment_C1 = false, increment_C2 = false, increment_C3 = false,
          decrement_C2 = false, update_coordinates = false;

        }  // Next iteration
      });  // End parallel for
    });    // End queue submit. End accessor scope
  }        // End buffer scope
}  // End of function ParticleMotion()
