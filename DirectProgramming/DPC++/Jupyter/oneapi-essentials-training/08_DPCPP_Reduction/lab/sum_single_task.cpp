//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 1024; // global size

int main() {
  //# setup sycl::queue with default device selector
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# user single_task to add all numbers
  q.single_task([=](){
    int sum = 0;
    for(int i=0;i<N;i++){
        sum += data[i];
    }
    data[0] = sum;
  }).wait();

  std::cout << "Sum = " << data[0] << "\n";
  
  free(data, q);
  return 0;
}
