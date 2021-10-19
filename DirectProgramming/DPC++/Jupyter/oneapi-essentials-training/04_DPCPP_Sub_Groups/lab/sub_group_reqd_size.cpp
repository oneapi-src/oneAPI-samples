//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64; // work-group size
static constexpr size_t S = 32; // sub_group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;
  
  auto sg_sizes = q.get_device().get_info<info::device::sub_group_sizes>();
  std::cout << "Supported Sub-Group Sizes : ";
  for (int i=0; i<sg_sizes.size(); i++) std::cout << sg_sizes[i] << " "; std::cout << std::endl;
    
  auto max_sg_size = std::max_element(sg_sizes.begin(), sg_sizes.end());
  std::cout << "Max Sub-Group Size        : " << max_sg_size[0] << std::endl;

  //# initialize data array using usm
  int *data = malloc_shared<int>(N, q);
  for(int i=0; i<N; i++) data[i] = i;

  //# use parallel_for and sub_groups
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item)[[intel::reqd_sub_group_size(S)]] {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# write sub_group tp zero except first location for each sub_group
    if (sg.get_local_id()[0] != 0) data[i] = 0;

  }).wait();

  for(int i=0; i<N; i++) std::cout << data[i] << " "; std::cout << std::endl;
  
  free(data, q);
  return 0;
}

