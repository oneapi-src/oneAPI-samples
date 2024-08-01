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
  const int N = 1200;

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
  auto gpus = sycl::platform(sycl::gpu_selector_v).get_devices();
  int num_devices = gpus.size();

  // Create sycl::queue for each gpu
  std::vector<sycl::queue> q(num_devices);
  for(int i = 0; i < num_devices; i++){
    std::cout << "Device: " << gpus[i].get_info<sycl::info::device::name>() << "\n";
    q.push_back(sycl::queue(gpus[i]));
  }

  // device mem alloc for vectors a,b,c for each device
  float *da[num_devices];
  float *db[num_devices];
  float *dc[num_devices];
  for (int i = 0; i < num_devices; i++) {
    da[i] = sycl::malloc_device<float>(N/num_devices, q[i]);
    db[i] = sycl::malloc_device<float>(N/num_devices, q[i]);
    dc[i] = sycl::malloc_device<float>(N/num_devices, q[i]);
  }

  // memcpy for matrix and b to device alloc
  for (int i = 0; i < num_devices; i++) {
    q[i].memcpy(&da[i][0], &a[i*N/num_devices], N/num_devices * sizeof(float));
    q[i].memcpy(&db[i][0], &b[i*N/num_devices], N/num_devices * sizeof(float));
  }

  // wait for copy to complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // submit vector-add kernels to all devices
  for (int i = 0; i < num_devices; i++)
    kernel_compute_vadd(q[i], da[i], db[i], dc[i], N/num_devices);

  // wait for compute complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // copy back result to host
  for (int i = 0; i < num_devices; i++)
    q[i].memcpy(&c[i*N/num_devices], &dc[i][0], N/num_devices * sizeof(float));

  // wait for copy to complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // print output
  for (int i = 0; i < N; i++) std::cout << c[i] << " ";
  std::cout << "\n";

  free(a);
  free(b);
  free(c);
  for (int i = 0; i < num_devices; i++) {
    sycl::free(da[i], q[i]);
    sycl::free(db[i], q[i]);
    sycl::free(dc[i], q[i]);
  }
  return 0;
}
