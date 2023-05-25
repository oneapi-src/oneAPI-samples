//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

static constexpr size_t N = 102400000; // global size

int main() {
  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
  //# setup queue with default selector
  sycl::queue q;
  std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  //# initialize data array using usm
  int *data = static_cast<int *>(malloc(N * sizeof(int)));
  for (int i = 0; i < N; i++) data[i] = 1;

  //# USM device allocation
  auto device_data = sycl::malloc_device<int>(N, q);

  //# copy mem from host to device
  q.memcpy(device_data, data, sizeof(int) * N).wait();

  //# single_task kernel performing simple addition of all elements
  q.single_task([=](){
    int sum = 0;
    for(int i=0;i<N;i++){
        sum += device_data[i];
    }
    device_data[0] = sum;
  }).wait();

  //# copy mem from device to host
  q.memcpy(data, device_data, sizeof(int) * N).wait();

  std::cout << "Sum = " << data[0] << "\n";
    
  auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
  std::cout << "Compute Duration: " << duration / 1e+9 << " seconds\n";

  sycl::free(device_data, q);
  free(data);
  return 0;
}
