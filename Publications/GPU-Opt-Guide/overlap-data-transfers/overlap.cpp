//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include <CL/sycl.hpp>

#define NITERS 10
#define KERNEL_ITERS 10000
#define NUM_CHUNKS 10
#define CHUNK_SIZE 10000000

class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

int main() {
  const int num_chunks = NUM_CHUNKS;
  const int chunk_size = CHUNK_SIZE;
  const int iter = NITERS;

  sycl::queue q;

  // Allocate and initialize host data
  float *host_data[num_chunks];
  for (int c = 0; c < num_chunks; c++) {
    host_data[c] = sycl::malloc_host<float>(chunk_size, q);
    float val = c;
    for (int i = 0; i < chunk_size; i++)
      host_data[c][i] = val;
  }
  std::cout << "Allocated host data\n";

  // Allocate and initialize device memory
  float *device_data[num_chunks];
  for (int c = 0; c < num_chunks; c++) {
    device_data[c] = sycl::malloc_device<float>(chunk_size, q);
    float val = 1000.0;
    q.fill<float>(device_data[c], val, chunk_size);
  }
  q.wait();
  std::cout << "Allocated device data\n";

  Timer timer;
  for (int it = 0; it < iter; it++) {
    for (int c = 0; c < num_chunks; c++) {
      auto add_one = [=](auto id) {
        for (int i = 0; i < KERNEL_ITERS; i++)
          device_data[c][id] += 1.0;
      };
      // Copy-in not dependent on previous event
      auto copy_in =
          q.memcpy(device_data[c], host_data[c], sizeof(float) * chunk_size);
      // Compute waits for copy_in
      auto compute = q.parallel_for(chunk_size, copy_in, add_one);
      auto cg = [=](auto &h) {
        h.depends_on(compute);
        h.memcpy(host_data[c], device_data[c], sizeof(float) * chunk_size);
      };
      // Copy out waits for compute
      auto copy_out = q.submit(cg);
    }

    q.wait();
  }
  auto elapsed = timer.Elapsed() / iter;
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
  std::cout << "Time = " << elapsed << " usecs\n";
}
// Snippet end
