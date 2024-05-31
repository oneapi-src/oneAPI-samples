//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <execution>

static constexpr size_t N = 5120; // global size
static constexpr size_t B = 32;   // WG size

void kernel_compute_mm(sycl::queue &q, float *a, float *b, float *c, size_t n, size_t wg) {
  q.parallel_for(
      sycl::nd_range<2>(sycl::range<2>{n, n}, sycl::range<2>{wg, wg}),
      [=](sycl::nd_item<2> item) {
        const int i = item.get_global_id(0);
        const int j = item.get_global_id(1);
        float temp = 0.0f;
        for (int k = 0; k < N; k++) {
          temp += a[i * N + k] * b[k * N + j];
        }
        c[i * N + j] = temp;
      });
}

int main() {
  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  // Define matrices
  auto matrix_a = static_cast<float *>(malloc(N * N * sizeof(float)));
  auto matrix_b = static_cast<float *>(malloc(N * N * sizeof(float)));
  auto matrix_c = static_cast<float *>(malloc(N * N * sizeof(float)));
  float v1 = 2.f;
  float v2 = 3.f;
  // Initialize matrices with values
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
        matrix_a[i * N + j] = v1++;
        matrix_b[i * N + j] = v2++;
        matrix_c[i * N + j] = 0.f;
    }

  // create queues for each device
  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // device mem alloc for matrix a,b,c for each device
  auto da = sycl::malloc_device<float>(N * N, q);
  auto db = sycl::malloc_device<float>(N * N, q);
  auto dc = sycl::malloc_device<float>(N * N, q);

  // memcpy for matrix and b to device alloc
  q.memcpy(da, matrix_a, N * N * sizeof(float));
  q.memcpy(db, matrix_b, N * N * sizeof(float));

  // wait for copy to complete
  q.wait();

  // submit matrix multiply kernels to all devices
  kernel_compute_mm(q, da, db, dc, N, B);

  // wait for compute complete
  q.wait();

  // copy back result to host
  q.memcpy(matrix_c, dc, N * N * sizeof(float)).wait();

  // print first element of result matrix
  std::cout << "\nMatrix Multiplication Complete\n\n";
    std::cout << "matrix_c[0][0]=" << matrix_c[0] << "\n";

  free(matrix_a);
  free(matrix_b);
  free(matrix_c);
  sycl::free(da, q);
  sycl::free(db, q);
  sycl::free(dc, q);

  auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
  std::cout << "\nCompute Duration: " << duration / 1e+9 << " seconds\n";
  return 0;
}
