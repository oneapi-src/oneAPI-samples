//
// Code example that uses OpenMP interop and SYCL kernel.
//
// The example shows
// 1) How to create an interop object to access the cooperating SYCL queue
// 2) How to check if the interop object is for SYCL foreign runtime
// 3) How to set up device accessible memory to be used in SYCL code
//
// Compilation command:
//   icpx -qopenmp -fopenmp-targets=spir64 -fsycl omp_interop_sycl_1.cpp
//
#include <cstdio>
#include <omp.h>
#include <sycl/sycl.hpp>

int main() {
  constexpr int num = 16384;
  float *a = new float[num];
  float *b = new float[num];
  float *c = new float[num];

  // Initialize host data.
  for (int i = 0; i < num; i++) {
    a[i] = i + 1;
    b[i] = 2 * i;
    c[i] = 0;
  }

  omp_interop_t obj{omp_interop_none};

  // Snippet begin0
  // Create an interop object with SYCL queue access.
#pragma omp interop init(prefer_type(omp_ifr_sycl), targetsync : obj)
  if (omp_ifr_sycl != omp_get_interop_int(obj, omp_ipr_fr_id, nullptr)) {
    fprintf(stderr, "ERROR: Failed to create interop with SYCL queue access\n");
    exit(1);
  }

  // Access SYCL queue returned by OpenMP interop.
  auto *q = static_cast<sycl::queue *>(
      omp_get_interop_ptr(obj, omp_ipr_targetsync, nullptr));
  // Snippet end0

  // Snippet begin1
  // Use OpenMP target data environment while allowing SYCL code to access the
  // device data with "use_device_ptr" clause.
#pragma omp target data map(to : a[0 : num], b[0 : num])                       \
    map(from : c[0 : num]) use_device_ptr(a, b, c)
  {
    auto event = q->parallel_for(num, [=](auto i) { c[i] = a[i] + b[i]; });
    event.wait();
  }

  // Release resources associated with "obj".
#pragma omp interop destroy(obj)
  // Snippet end1

  printf("c[0] = %.3f (%.3f), c[%d] = %.3f (%.3f)\n", c[0], 1.0, num - 1,
         c[num - 1], 3.0 * (num - 1) + 1);
  delete[] a;
  delete[] b;
  delete[] c;

  return 0;
}
