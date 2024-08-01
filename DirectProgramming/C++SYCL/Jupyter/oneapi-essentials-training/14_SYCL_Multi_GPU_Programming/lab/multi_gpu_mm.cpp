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

  // find all GPUs
  auto gpus = sycl::platform(sycl::gpu_selector_v).get_devices();
  int num_devices = gpus.size();

  // Define matrices
  float *matrix_a[num_devices];
  float *matrix_b[num_devices];
  float *matrix_c[num_devices];

  float v1 = 2.f;
  float v2 = 3.f;
  for (int n = 0; n < num_devices; n++) {
    matrix_a[n] = static_cast<float *>(malloc(N * N * sizeof(float)));
    matrix_b[n] = static_cast<float *>(malloc(N * N * sizeof(float)));
    matrix_c[n] = static_cast<float *>(malloc(N * N * sizeof(float)));

    // Initialize matrices with values
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        matrix_a[n][i * N + j] = v1++;
        matrix_b[n][i * N + j] = v2++;
        matrix_c[n][i * N + j] = 0.f;
      }
  }

  float *da[num_devices];
  float *db[num_devices];
  float *dc[num_devices];

  std::vector<sycl::queue> q(num_devices);
  std::vector<int> id;

  // create queues for each device
  std::cout << "\nSubmitting Compute Kernel to GPUs:\n";
  for (int i = 0; i < num_devices; i++) {
    q[i] = sycl::queue(gpus[i]);
    id.push_back(i);
    std::cout << "GPU" << i << ": " << q[i].get_device().get_info<sycl::info::device::name>() << "\n";
  }

  // device mem alloc for matrix a,b,c for each device
  for (int i = 0; i < num_devices; i++) {
    da[i] = sycl::malloc_device<float>(N * N, q[i]);
    db[i] = sycl::malloc_device<float>(N * N, q[i]);
    dc[i] = sycl::malloc_device<float>(N * N, q[i]);
  }

  // memcpy for matrix and b to device alloc
  for (int i = 0; i < num_devices; i++) {
    q[i].memcpy(&da[i][0], &matrix_a[i][0], N * N * sizeof(float));
    q[i].memcpy(&db[i][0], &matrix_b[i][0], N * N * sizeof(float));
  }

  // wait for copy to complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // submit matrix multiply kernels to all devices
  /*
  for (int i = 0; i < num_devices; i++)
    kernel_compute_mm(q[i], da[i], db[i], dc[i], N, B);
  */
  std::for_each(std::execution::par, id.begin(), id.end(), [&q, &da, &db, &dc](auto i){
    kernel_compute_mm(q[i], da[i], db[i], dc[i], N, B);
  });

  // wait for compute complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // copy back result to host
  for (int i = 0; i < num_devices; i++)
    q[i].memcpy(&matrix_c[i][0], &dc[i][0], N * N * sizeof(float));

  // wait for copy to complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // print first element of result matrix
  std::cout << "\nMatrix Multiplication Complete\n\n";
  for (int i = 0; i < num_devices; i++)
    std::cout << "GPU" << i << ": matrix_c[0][0]=" << matrix_c[i][0] << "\n";

  for (int i = 0; i < num_devices; i++) {
    free(matrix_a[i]);
    free(matrix_b[i]);
    free(matrix_c[i]);
    sycl::free(da[i], q[i]);
    sycl::free(db[i], q[i]);
    sycl::free(dc[i], q[i]);
  }

  auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
  std::cout << "\nCompute Duration: " << duration / 1e+9 << " seconds\n";
  return 0;
}
