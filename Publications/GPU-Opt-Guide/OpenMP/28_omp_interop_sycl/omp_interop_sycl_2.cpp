//
// Code example similar to the one in the OpenMP API 5.2.2 Examples document.
// The example uses SYCL foreign runtime and MKL.
//
// The example shows
// 1) How to initialize an interop object with SYCL queue access
// 2) How to check if the interop object is for SYCL foreign runtime
// 3) How to allocate SYCL memory object from the interop
// 4) How to use the interop object with the dispatch construct and MKL
//
// Compilation command (requires MKL):
//   icpx -qopenmp -fopenmp-targets=spir64 -fsycl -qmkl omp_interop_sycl_2.cpp
//
#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <mkl_omp_offload.h>
#include <omp.h>
#include <sycl/sycl.hpp>

#define N 16384

void myVectorSet(int n, float s, float *x) {
  for (int i = 0; i < n; ++i)
    x[i] = s * (i + 1);
}

void mySscal(int n, float s, float *x) {
  for (int i = 0; i < n; ++i)
    x[i] = s * x[i];
}

int main() {
  const float scalar = 2.0;
  float *x, *y, *d_x, *d_y;
  int dev;

  omp_interop_t obj{omp_interop_none};
  dev = omp_get_default_device();

  // Snippet begin0
  // Create an interop object with SYCL queue access
#pragma omp interop init(prefer_type(omp_ifr_sycl), targetsync : obj)          \
    device(dev)

  if (omp_ifr_sycl != omp_get_interop_int(obj, omp_ipr_fr_id, nullptr)) {
    fprintf(stderr, "ERROR: Failed to create interop with SYCL queue access\n");
    exit(1);
  }
  sycl::queue *q = static_cast<sycl::queue *>(
      omp_get_interop_ptr(obj, omp_ipr_targetsync, nullptr));

  // Allocate host data.
  x = new float[N];
  y = new float[N];

  // Allocate device data using SYCL queue.
  d_x = sycl::malloc_device<float>(N * sizeof(float), *q);
  d_y = sycl::malloc_device<float>(N * sizeof(float), *q);
  // Snippet end0

  // Snippet begin1
  // Associate device pointers with host pointers
  omp_target_associate_ptr(&x[0], d_x, N * sizeof(float), 0, dev);
  omp_target_associate_ptr(&y[0], d_y, N * sizeof(float), 0, dev);
  // Snippet end1

  // Snippet begin2
  // Initialize host data.
  myVectorSet(N, 1.0, x);
  myVectorSet(N, -1.0, y);

  // Perform SYCL's memory copy operations from host to device.
  q->memcpy(d_x, x, N * sizeof(float));
  q->memcpy(d_y, y, N * sizeof(float));
  q->wait();

  // Invoke MKL's variant function using the dispatch construct, appending the
  // interop object and replacing the host pointers with device pointers for
  // "x" and "y" as specified in the directives used in MKL.
#pragma omp dispatch interop(obj)
  cblas_saxpy(N, scalar, x, 1, y, 1);

  // Perform SYCL's memory copy operation from device to host.
  q->memcpy(y, d_y, N * sizeof(float));
  q->wait();
  // Snippet end2

  // Snippet begin3
  // Update device data for "x" and bring them back to host.
#pragma omp target map(always, from : x[0 : N])
  mySscal(N, scalar, x);

  printf("(1:16384) %.3f:%.3f\n", y[0], y[N - 1]);
  printf("(2:32768) %.3f:%.3f\n", x[0], x[N - 1]);

  // Remove the associated device data for the host pointers.
  omp_target_disassociate_ptr(&x[0], dev);
  omp_target_disassociate_ptr(&y[0], dev);

  delete[] x;
  delete[] y;
  sycl::free(d_x, *q);
  sycl::free(d_y, *q);

#pragma omp interop destroy(obj)
  // Snippet end3

  return 0;
}
