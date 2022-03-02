// Allocate Shared Memory
float *x =
    (float *)omp_target_alloc_shared(ARRAY_SIZE * sizeof(float), deviceId);
float *y =
    (float *)omp_target_alloc_shared(ARRAY_SIZE * sizeof(float), deviceId);