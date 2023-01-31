//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  int *data = malloc_shared<int>(N, q);
  for(int i=0; i<N; i++) data[i] = i;
  for(int i=0; i<N; i++) std::cout << data[i] << " "; 
  std::cout << "\n";  

  //# use parallel_for and sub_groups
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# write sub_group item values to broadcast value at index 3
    data[i] = group_broadcast(sg, data[i], 3);

  }).wait();

  for(int i=0; i<N; i++) std::cout << data[i] << " "; 
  std::cout << "\n";
  
  free(data, q);
  return 0;
}

