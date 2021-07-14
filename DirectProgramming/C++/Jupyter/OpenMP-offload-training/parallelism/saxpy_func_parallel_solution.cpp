#pragma omp target map(from: is_cpu) map(from:num_teams) map(to:x[0:ARRAY_SIZE]) map(tofrom:y[0:ARRAY_SIZE])
{
  // 1. Add pragma to create multiple master threads use clause num_teams(NUM_BLOCKS)
  //    and distribute loop iterations to the various master threads.
#pragma omp teams distribute num_teams(NUM_BLOCKS)
  for (ib = 0; ib < ARRAY_SIZE; ib += NUM_BLOCKS) {
    if (ib == 0) {
      // Test if target is the CPU Host or the GPU Device
      is_cpu = omp_is_initial_device();
      // Query number of teams created
      num_teams = omp_get_num_teams();
    }

    // 2. Place the combined pragma here to create a team of threads for each master thread
    //   Distribute iterations to those threads, and vectorize
#pragma omp parallel for simd
    for (i = ib; i < ib + NUM_BLOCKS; i++) {
      y[i] = a * x[i] + y[i];
    }
  }
}