#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N_A 1000000
#define N_B 500000
#define NUM_ITERATIONS 10000

int main() {
  float *a = static_cast<float *>(malloc(N_A * sizeof(float)));
  float *b = static_cast<float *>(malloc(N_B * sizeof(float)));
  float *res = static_cast<float *>(malloc(N_A * sizeof(float)));

  // Initialize a and b
  for (int i = 0; i < N_A; ++i)
    a[i] = i * 0.0001f;
  for (int i = 0; i < N_B; ++i)
    b[i] = (i % 1000) * 0.001f;

  float sum = 0;
  int count = 0;
  double start, end;

// Dummy target region to warm up GPU
#pragma omp target
  {
    ;
  }

  start = omp_get_wtime();
  for (int j = 0; j < NUM_ITERATIONS; ++j) {
#pragma omp target teams distribute parallel for map(to : a[0 : N_A])          \
    map(from : res[0 : N_A])
    for (int i = 0; i < N_A; ++i) {
      float acc = a[i];
      for (int k = 0; k < 20; ++k)
        acc = sinf(acc) * expf(acc) + acc * 1.01f;
      res[i] = acc;
    }
#pragma omp target teams distribute parallel for map(to : b[0 : N_B])          \
    reduction(+ : sum, count)
    for (int i = 0; i < N_B; ++i) {
      float val = b[i];
      if (val > 0.5f)
        count++;
      sum += val * 0.1f;
    }
  }
  end = omp_get_wtime();
  printf("GPU only time: %f seconds.\n", end - start);
  free(a);
  free(b);
  free(res);
  return 0;
}
