//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include<dpstd/execution>
#include<dpstd/algorithm>
using namespace sycl;
using namespace dpstd::execution;
const int N = 4;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;
    
  //# USM allocation on device
  int* data = malloc_shared<int>(N, q);
    
  //# Parallel STL algorithm using USM pointer
  std::fill(make_device_policy(q), data, data + N, 20);
  q.wait();
    
  for (int i = 0; i < N; i++) std::cout << data[i] << std::endl;
  free(data, q);
  return 0;
}
