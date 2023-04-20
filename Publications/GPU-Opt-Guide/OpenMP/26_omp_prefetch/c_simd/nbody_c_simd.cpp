#include <chrono>
#include <math.h>
#include <omp.h>
#include <stdio.h>

#define CACHE_CLEAN_SIZE 100000000
#define ITERATIONS 100
#define ARRAYLEN1 4096
#define ARRAYLEN2 32768
#define VECLEN 16
// snippet-begin
#define WORKGROUP_SIZE 1024
#define PREFETCH_HINT 4 // 4 = prefetch to L1 and L3;  2 = prefetch to L3
#define TILE_SIZE 64

void nbody_1d_gpu(float *c, float *a, float *b, int n1, int n2) {
#pragma omp target teams distribute parallel for thread_limit(WORKGROUP_SIZE / \
                                                              VECLEN)
  for (int i = 0; i < n1; i += VECLEN) {
    const float ma0 = 0.269327f, ma1 = -0.0750978f, ma2 = 0.0114808f;
    const float ma3 = -0.00109313f, ma4 = 0.0000605491f, ma5 = -0.00000147177f;
    const float eps = 0.01f;

    float dx[VECLEN];
    float aa[VECLEN], bb[TILE_SIZE];
#pragma omp simd simdlen(VECLEN)
#pragma unroll(0)
    for (int v = 0; v < VECLEN; ++v) {
      dx[v] = 0.0f;
      aa[v] = a[i + v];
    }
    for (int j = 0; j < n2; j += TILE_SIZE) {
      // load tile from b
      for (int u = 0; u < TILE_SIZE; u += VECLEN) {
#pragma omp simd simdlen(VECLEN)
#pragma unroll(0)
        for (int v = 0; v < VECLEN; ++v)
          bb[u + v] = b[j + u + v];
#ifdef PREFETCH
        int next_tile = j + TILE_SIZE + u;
#pragma ompx prefetch data(PREFETCH_HINT : b[next_tile]) if (next_tile < n2)
#endif
      }
// compute current tile
#pragma omp simd simdlen(VECLEN)
#pragma unroll(0)
      for (int v = 0; v < VECLEN; ++v) {
#pragma unroll(TILE_SIZE)
        for (int u = 0; u < TILE_SIZE; ++u) {
          float delta = bb[u] - aa[v];
          float r2 = delta * delta;
          float s0 = r2 + eps;
          float s1 = 1.0f / sqrtf(s0);
          float f =
              (s1 * s1 * s1) -
              (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + ma5)))));
          dx[v] += f * delta;
        }
      }
    }
#pragma omp simd simdlen(VECLEN)
#pragma unroll(0)
    for (int v = 0; v < VECLEN; ++v) {
      c[i + v] = dx[v] * 0.23f;
    }
  }
}
// snippet-end

void nbody_1d_cpu(float *c, float *a, float *b, int n1, int n2) {
  for (int i = 0; i < n1; ++i) {
    const float ma0 = 0.269327f, ma1 = -0.0750978f, ma2 = 0.0114808f;
    const float ma3 = -0.00109313f, ma4 = 0.0000605491f, ma5 = -0.00000147177f;
    const float eps = 0.01f;

    float dx = 0.0f;
    for (int j = 0; j < n2; ++j) {
      float delta = b[j] - a[i];
      float r2 = delta * delta;
      float s0 = r2 + eps;
      float s1 = 1.0f / sqrtf(s0);
      float f = (s1 * s1 * s1) -
                (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + ma5)))));
      dx += f * delta;
    }
    c[i] = dx * 0.23f;
  }
}

void clean_cache_gpu(double *d, int n) {

#pragma omp target teams distribute parallel for thread_limit(1024)
  for (unsigned i = 0; i < n; ++i)
    d[i] = i;

  return;
}

int main() {

  float *a, *b, *c;
  double *d;

  a = new float[ARRAYLEN1];
  b = new float[ARRAYLEN2];
  c = new float[ARRAYLEN1];
  d = new double[CACHE_CLEAN_SIZE];

  // intialize
  float dx = 1.0f / (float)ARRAYLEN2;
  b[0] = 0.0f;
  for (int i = 1; i < ARRAYLEN2; ++i) {
    b[i] = b[i - 1] + dx;
  }
  for (int i = 0; i < ARRAYLEN1; ++i) {
    a[i] = b[i];
    c[i] = 0.0f;
  }

#pragma omp target
  {}

#pragma omp target enter data map(alloc                                        \
                                  : a [0:ARRAYLEN1], b [0:ARRAYLEN2],          \
                                    c [0:ARRAYLEN1])
#pragma omp target enter data map(alloc : d [0:CACHE_CLEAN_SIZE])

#pragma omp target update to(a [0:ARRAYLEN1], b [0:ARRAYLEN2])

  double t1, t2, elapsed_s = 0.0;
  for (int i = 0; i < ITERATIONS; ++i) {
    clean_cache_gpu(d, CACHE_CLEAN_SIZE);

    t1 = omp_get_wtime();
    nbody_1d_gpu(c, a, b, ARRAYLEN1, ARRAYLEN2);
    t2 = omp_get_wtime();

    elapsed_s += (t2 - t1);
  }

#pragma omp target update from(c [0:ARRAYLEN1])

  float sum = 0.0f;
  for (int i = 0; i < ARRAYLEN1; ++i)
    sum += c[i];
  printf("Obtained output = %8.3f\n", sum);

  for (int i = 0; i < ARRAYLEN1; ++i)
    c[i] = 0.0f;
  nbody_1d_cpu(c, a, b, ARRAYLEN1, ARRAYLEN2);
  sum = 0.0f;
  for (int i = 0; i < ARRAYLEN1; ++i)
    sum += c[i];
  printf("Expected output = %8.3f\n", sum);

  printf("\nTotal time = %8.1f milliseconds\n", (elapsed_s * 1000));

#pragma omp target exit data map(delete                                        \
                                 : a [0:ARRAYLEN1], b [0:ARRAYLEN2],           \
                                   c [0:ARRAYLEN1])
#pragma omp target exit data map(delete : d [0:CACHE_CLEAN_SIZE])

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;

  return 0;
}
