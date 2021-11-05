// Add the target pragma with the map clauses here
#pragma omp target map(tofrom: y[0:ARRAY_SIZE]) map(to: x[0:ARRAY_SIZE]) map(from: is_cpu)
{
    is_cpu = omp_is_initial_device();
    for (i = 0; i < ARRAY_SIZE; i++) {
        y[i] = a * x[i] + y[i];
    }
}
