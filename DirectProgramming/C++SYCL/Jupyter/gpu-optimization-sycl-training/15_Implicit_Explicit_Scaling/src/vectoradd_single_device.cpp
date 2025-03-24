//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

void kernel_compute_vadd(sycl::queue &q, float *a, float *b, float *c, size_t n) {
  q.parallel_for(n, [=](auto i) {
    c[i] = a[i] + b[i];
  });
}

int main() {
  const int N = 1680;

  // Define 3 arrays
  float *a = static_cast<float *>(malloc(N * sizeof(float)));
  float *b = static_cast<float *>(malloc(N * sizeof(float)));
  float *c = static_cast<float *>(malloc(N * sizeof(float)));

  // Initialize matrices with values
  for (int i = 0; i < N; i++){
    a[i] = 1;
    b[i] = 2;
    c[i] = 0;
  }
    
  // get all GPUs devices into a vector
  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "-EUs  : " << q.get_device().get_info<sycl::info::device::max_compute_units>() << "\n";

  // device mem alloc for vectors a,b,c for device
  auto da = sycl::malloc_device<float>(N, q);
  auto db = sycl::malloc_device<float>(N, q);
  auto dc = sycl::malloc_device<float>(N, q);

  // memcpy for matrix and b to device alloc
  q.memcpy(da, a, N * sizeof(float));
  q.memcpy(db, b, N * sizeof(float));
  q.wait();

  kernel_compute_vadd(q, da, db, dc, N);
  q.wait();

  // copy back result to host
  q.memcpy(c, dc, N * sizeof(float));
  q.wait();

  // print output
  for (int i = 0; i < N; i++) std::cout << c[i] << " ";
  std::cout << "\n";

  free(a);
  free(b);
  free(c);
  sycl::free(da, q);
  sycl::free(db, q);
  sycl::free(dc, q);
  return 0;
}
