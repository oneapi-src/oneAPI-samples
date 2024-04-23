//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

#define NITERS 10
#define KERNEL_ITERS 10000
#define NUM_CHUNKS 10
#define CHUNK_SIZE 10000000

int main() {
  const int num_chunks = NUM_CHUNKS;
  const int chunk_size = CHUNK_SIZE;
  const int iter = NITERS;

  sycl::queue q;

  //# Allocate and initialize host data
  float *host_data[num_chunks];
  for (int c = 0; c < num_chunks; c++) {
    host_data[c] = sycl::malloc_host<float>(chunk_size, q);
    float val = c;
    for (int i = 0; i < chunk_size; i++)
      host_data[c][i] = val;
  }
  std::cout << "Allocated host data\n";

  //# Allocate and initialize device memory
  float *device_data[num_chunks];
  for (int c = 0; c < num_chunks; c++) {
    device_data[c] = sycl::malloc_device<float>(chunk_size, q);
    float val = 1000.0;
    q.fill<float>(device_data[c], val, chunk_size);
  }
  q.wait();
  std::cout << "Allocated device data\n";

  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  for (int it = 0; it < iter; it++) {
    for (int c = 0; c < num_chunks; c++) {

      //# Copy-in not dependent on previous event
      auto copy_in_event = q.memcpy(device_data[c], host_data[c], sizeof(float) * chunk_size);

      //# Compute waits for copy_in_event
      auto compute_event = q.parallel_for(chunk_size, copy_in_event, [=](auto id) {
        for (int i = 0; i < KERNEL_ITERS; i++) device_data[c][id] += 1.0;
      });

      //# Copy out waits for compute_event
      auto copy_out_event = q.memcpy(host_data[c], device_data[c], sizeof(float) * chunk_size, compute_event);
    }

    q.wait();
  }
  auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;

  for (int c = 0; c < num_chunks; c++) {
    for (int i = 0; i < chunk_size; i++) {
      if (host_data[c][i] != (float)((c + KERNEL_ITERS * iter))) {
        std::cout << "Mismatch for chunk: " << c << " position: " << i
                  << " expected: " << c + 10000 << " got: " << host_data[c][i]
                  << "\n";
        break;
      }
    }
  }

  std::cout << "Compute Duration: " << duration / 1e+9 << " seconds\n";
}
