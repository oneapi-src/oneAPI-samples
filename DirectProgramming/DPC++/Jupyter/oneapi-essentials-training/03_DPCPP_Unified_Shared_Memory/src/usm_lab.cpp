//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
using namespace sycl;

static const int N = 1024;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //intialize 2 arrays on host
  int *data1 = static_cast<int *>(malloc(N * sizeof(int)));
  int *data2 = static_cast<int *>(malloc(N * sizeof(int)));
  for (int i = 0; i < N; i++) {
    data1[i] = 25;
    data2[i] = 49;
  }
    
  //# STEP 1 : Create USM device allocation for data1 and data2

  auto data1_device = malloc_device<int>(N, q);
  auto data2_device = malloc_device<int>(N, q);
    
  //# STEP 2 : Copy data1 and data2 to USM device allocation
    
  auto e1 = q.memcpy(data1_device, data1, N * sizeof(int));
  auto e2 = q.memcpy(data2_device, data2, N * sizeof(int));
    
  //# STEP 3 : Write kernel code to update data1 on device with sqrt of value
    
  auto e3 = q.parallel_for(N, e1, [=](auto i) { 
    data1_device[i] = sqrt(data1_device[i]);
  });

  //# STEP 4 : Write kernel code to update data2 on device with sqrt of value
    
  auto e4 = q.parallel_for(N, e2, [=](auto i) { 
    data2_device[i] = sqrt(data2_device[i]); 
  });

  //# STEP 5 : Write kernel code to add data2 on device to data1

  auto e5 = q.parallel_for(N, {e3,e4}, [=](auto i) { data1_device[i] += data2_device[i]; });

  //# STEP 6 : Copy result from device to data1
    
  q.memcpy(data1, data1_device, N * sizeof(int), e5).wait();

  //# verify results
  int fail = 0;
  for (int i = 0; i < N; i++) if(data1[i] != 12) {fail = 1; break;}
  if(fail == 1) std::cout << " FAIL"; else std::cout << " PASS";
  std::cout << "\n";

  free(data1_device, q);
  free(data2_device, q);
  return 0;
}
