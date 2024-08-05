//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main() {
  // get all GPUs devices into a vector
  auto gpus = sycl::platform(sycl::gpu_selector_v).get_devices();

  if(gpus.size() >= 2) {
    // Initialize array with values
    const int N = 256;
    float a[N], b[N];
    for (int i = 0; i < N; i++){
      a[i] = i;
    }
      
    // Create sycl::queue for each gpu
    sycl::queue q0(gpus[0]);
    std::cout << "GPU0: " << gpus[0].get_info<sycl::info::device::name>() << "\n";
    sycl::queue q1(gpus[1]);
    std::cout << "GPU1: " << gpus[1].get_info<sycl::info::device::name>() << "\n";

    // device mem alloc for each device
    auto a0 = sycl::malloc_device<float>(N, q0);
    auto a1 = sycl::malloc_device<float>(N, q1);

    // memcpy to device alloc
    q0.memcpy(a0, a, N * sizeof(float));
    q1.memcpy(a1, a, N * sizeof(float));

    // wait for copy to complete
    q0.wait();
    q1.wait();

    // submit kernels to 2 devices
    q0.parallel_for(N, [=](auto i) {
      a0[i] *= 2;
    });
    q1.parallel_for(N, [=](auto i) {
      a1[i] *= 3;
    });

    // wait for compute complete
    q0.wait();
    q1.wait();

    // copy back result to host
    q0.memcpy(a, a0, N * sizeof(float));
    q1.memcpy(b, a1, N * sizeof(float));

    // wait for copy to complete
    q0.wait();
    q1.wait();

    // print output
    for (int i = 0; i < N; i++) std::cout << a[i] << " ";
    std::cout << "\n";
    for (int i = 0; i < N; i++) std::cout << b[i] << " ";
    std::cout << "\n";

    sycl::free(a0, q0);
    sycl::free(a1, q1);
  } else {
      std::cout << "Multiple GPUs not available\n";
  }
  return 0;
}
