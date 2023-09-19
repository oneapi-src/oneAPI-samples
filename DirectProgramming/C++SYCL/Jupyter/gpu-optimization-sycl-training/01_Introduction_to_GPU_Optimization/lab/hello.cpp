//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main(){
  // Create SYCL queue
  sycl::queue q;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Allocate memory
  const int N = 16;
  auto data = sycl::malloc_shared<int>(N, q);
  for (int i=0; i<N; i++)data[i] = i;

  // Submit kernel to device
  q.parallel_for(N, [=](auto i){
    data[i] *= 5;
  }).wait();

  // Print output
  for (int i=0; i<N; i++) std::cout << data[i] << " "; 
  std::cout << "\n";

  sycl::free(data, q);
  return 0;
}