#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <sycl/sycl.hpp>

// Snippet begin0
// OpenMP saxpy
void saxpy_omp(float a, float *x, float *y, size_t n) {
#pragma omp target teams distribute parallel for map(to : x[0 : n])            \
    map(tofrom : y[0 : n])
  for (size_t i = 0; i < n; i++)
    y[i] = a * x[i] + y[i];
}
// Snippet end0

// Snippet begin1
// SYCL saxpy
void saxpy_sycl(float a, float *x, float *y, size_t n, sycl::queue &q) {
  size_t data_size = n * sizeof(float);
  float *d_x = sycl::malloc_device<float>(data_size, q);
  float *d_y = sycl::malloc_device<float>(data_size, q);
  q.memcpy(d_x, x, data_size);
  q.memcpy(d_y, y, data_size);
  q.wait();
  q.parallel_for(n, [=](auto i) { d_y[i] = a * d_x[i] + d_y[i]; });
  q.wait();
  q.memcpy(y, d_y, data_size);
  q.wait();
  sycl::free(d_x, q);
  sycl::free(d_y, q);
}
// Snippet end1

// Snippet begin2
int main() {
  constexpr size_t num = (64 << 10);
  constexpr size_t repeat = 1000;

  omp_interop_t obj{omp_interop_none};

#pragma omp interop init(prefer_type(omp_ifr_sycl), targetsync : obj)

  if (omp_ifr_sycl != omp_get_interop_int(obj, omp_ipr_fr_id, nullptr)) {
    printf("ERROR: Cannot access SYCL queue with OpenMP interop\n");
    return EXIT_FAILURE;
  }

  sycl::queue *q = static_cast<sycl::queue *>(
      omp_get_interop_ptr(obj, omp_ipr_targetsync, nullptr));

  float *x = new float[num];
  float *y = new float[num];

  auto init = [=]() {
    for (auto i = 0; i < num; i++) {
      x[i] = i + 1;
      y[i] = i;
    }
  };

  saxpy_omp(3, x, y, num); // Warm up
  init();
  double omp_sec = omp_get_wtime();
  for (size_t i = 0; i < repeat; i++)
    saxpy_omp(3, x, y, num);
  omp_sec = omp_get_wtime() - omp_sec;
  printf("OpenMP y[%d] = %.3e, y[%zu] = %.3e\n", 0, y[0], num - 1, y[num - 1]);

  saxpy_sycl(3, x, y, num, *q); // Warm up
  init();
  double sycl_sec = omp_get_wtime();
  for (size_t i = 0; i < repeat; i++)
    saxpy_sycl(3, x, y, num, *q);
  sycl_sec = omp_get_wtime() - sycl_sec;
  printf("  SYCL y[%d] = %.3e, y[%zu] = %.3e\n", 0, y[0], num - 1, y[num - 1]);

  printf("OpenMP took %.3f msec\n", omp_sec * 1e3);
  printf("  SYCL took %.3f msec\n", sycl_sec * 1e3);

  delete[] x;
  delete[] y;

#pragma omp interop destroy(obj)

  return EXIT_SUCCESS;
}
// Snippet end2
