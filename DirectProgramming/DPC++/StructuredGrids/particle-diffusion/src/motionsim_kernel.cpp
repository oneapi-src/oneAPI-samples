//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// motionsim_kernel:
// Intel® oneAPI DPC++ Language Basics Using a Monte Carlo
// Simulation (Device)
//

// Current and previous cell coordinates
#define CURR_COORDINATES iX + iY* grid_size
#define PREV_COORDINATES_ARRAY \
  prev_known_cell_coordinate_X + prev_known_cell_coordinate_Y* grid_size

// This function distributes simulation work across workers
void ParticleMotion(queue& q, const int seed, float* particle_X,
                    float* particle_Y, float* random_X, float* random_Y,
                    size_t* grid, const size_t grid_size, const size_t planes,
                    const size_t n_particles, const size_t n_iterations,
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
  mkl::rng::philox4x32x10 engine(q, seed);
  // Distribution object
  mkl::rng::gaussian<float, mkl::rng::gaussian_method::icdf> distr(ALPHA,
                                                                   SIGMA);
  // Begin buffer scope
  {
    // Create buffers using DPC++ buffer class
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
        // Particle number (used for indexing)
        size_t p = item.get_id(0);
        // Current coordinates of the particle
        int iX, iY;
        // Coordinates of the last known cell this particle resided in
        unsigned int prev_known_cell_coordinate_X, prev_known_cell_coordinate_Y;
        // True if particle is numerically within the cell radius
        bool within_radius;
        // True when particle is found to be in a cell
        bool inside_cell = false;
        // Particle displacements
        float displacement_X = 0.0f, displacement_Y = 0.0f;
        // Atomic operations flags
        bool increment_C1 = false, increment_C2 = false, increment_C3 = false,
             decrement_C2_for_previous_cell = false, update_coordinates = false;
        // Index variable for 3rd dimension of grid
        size_t layer = 0;

        // --Start iterations--
        // Each iteration:
        //    1. Updates the position of all particles
        //    2. Checks if particle is inside a cell or not
        //    3. Updates counters in cells array (grid)

        // Each particle performs this loop
        for (size_t iter = 0; iter < n_iterations; ++iter) {
          // Set the displacements to the random numbers
          displacement_X = random_X_a[iter * n_particles + p];
          displacement_Y = random_Y_a[iter * n_particles + p];

          // Displace particles
          particle_X_a[p] += displacement_X, particle_Y_a[p] += displacement_Y;

          // Compute distances from particle position to grid point i.e.,
          // the particle's distance from center of cell. Subtract the
          // integer value from floating point value to get just the
          // decimal portion. Use this value to later determine if the
          // particle is in the cell or outside the cell
          float dX = particle_X_a[p] - sycl::trunc(particle_X_a[p]);
          float dY = particle_Y_a[p] - sycl::trunc(particle_Y_a[p]);

          // Compute grid point indices
          iX = sycl::floor(particle_X_a[p]), iY = sycl::floor(particle_Y_a[p]);

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
                       --No action.
          */

          // Check if particle is still in computation grid
          if ((particle_X_a[p] < grid_size) && (particle_Y_a[p] < grid_size) &&
              (particle_X_a[p] >= 0) && (particle_Y_a[p] >= 0)) {
            // Compare radius to particle's distance from center of cell
            within_radius =
                radius >= sycl::sqrt(dX * dX + dY * dY) ? true : false;

            // Check if particle is in cell
            if (within_radius) {
              // Satisfies counter 1 requirement for cases 1, 3, 4
              increment_C1 = true;
              // Case 1
              if (!inside_cell)
                increment_C2 = true, increment_C3 = true, inside_cell = true,
                update_coordinates = true;
              // Case 3
              else if (prev_known_cell_coordinate_X != iX ||
                       prev_known_cell_coordinate_Y != iY)
                decrement_C2_for_previous_cell = true, increment_C2 = true,
                increment_C3 = true, update_coordinates = true;
              // Else: Case 4

            }  // End inside cell if statement

            // Case 2a --Particle remained inside grid and moved outside cell
            else if (inside_cell)
              inside_cell = false, decrement_C2_for_previous_cell = true;
            // Else: Case 5a --Particle remained inside grid and outside cell

          }  // End inside grid if statement

          // Case 2b --Particle moved outside grid and outside cell
          else if (inside_cell)
            inside_cell = false, decrement_C2_for_previous_cell = true;
          // Else: Case 5b --Particle remained outside of grid

          if (update_coordinates)
            prev_known_cell_coordinate_X = iX,
            prev_known_cell_coordinate_Y = iY;

          // Counter 1 layer of the grid (0 * grid_size * grid_size)
          layer = 0;
          if (increment_C1)
            atomic_fetch_add<size_t>(grid_a[CURR_COORDINATES + layer], 1);

          // Counter 2 layer of the grid (1 * grid_size * grid_size)
          layer = gs2;
          if (increment_C2)
            atomic_fetch_add<size_t>(grid_a[CURR_COORDINATES + layer], 1);

          // Counter 3 layer of the grid (2 * grid_size * grid_size)
          layer = gs2 + gs2;
          if (increment_C3)
            atomic_fetch_add<size_t>(grid_a[CURR_COORDINATES + layer], 1);

          // Counter 2 layer of the grid (1 * grid_size * grid_size)
          layer = gs2;
          if (decrement_C2_for_previous_cell)
            atomic_fetch_sub<size_t>(grid_a[PREV_COORDINATES_ARRAY + layer], 1);

          increment_C1 = false, increment_C2 = false, increment_C3 = false,
          decrement_C2_for_previous_cell = false, update_coordinates = false;

        }  // Next iteration
      });  // End parallel for
    });    // End queue submit. End accessor scope
  }        // End buffer scope
}  // End of function ParticleMotion()
