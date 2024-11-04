#include <omp.h>
#include <stdio.h>
#include <sycl/sycl.hpp>

#define EXTERN_C extern "C"

EXTERN_C void foo_(int *c, int *v1, int *n) {
  printf("ERROR: Base function foo should not be called\n");
}

EXTERN_C void foo_gpu_(int *c, int *v1, int *n, omp_interop_t obj) {
  int c_val = *c;
  int n_val = *n;

  if (omp_ifr_sycl != omp_get_interop_int(obj, omp_ipr_fr_id, nullptr)) {
    printf("Compute on host\n");
    for (int i = 0; i < n_val; i++)
      v1[i] = c_val * v1[i];
    return;
  }

  auto *q = static_cast<sycl::queue *>(
      omp_get_interop_ptr(obj, omp_ipr_targetsync, nullptr));

  printf("Compute on device\n");
#pragma omp target data map(tofrom : v1[0 : n_val]) use_device_ptr(v1)
  q->parallel_for(n_val, [=](auto i) { v1[i] = c_val * v1[i]; });
  q->wait();
}
