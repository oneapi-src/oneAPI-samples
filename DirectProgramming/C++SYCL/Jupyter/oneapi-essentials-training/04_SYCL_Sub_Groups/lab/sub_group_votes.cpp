//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 32; // global size
static constexpr size_t B = 16; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize input and output array using usm
  auto input = malloc_shared<int>(N, q);
  auto all = malloc_shared<int>(N, q);
  auto any = malloc_shared<int>(N, q);
  auto none = malloc_shared<int>(N, q);
    
  //# initialize values for input array  
  for(int i=0; i<N; i++) { if (i< 10) input[i] = 0; else input[i] = i; }
  std::cout << "input:\n";
  for(int i=0; i<N; i++) std::cout << input[i] << " "; std::cout << "\n";  

  //# use parallel_for and sub_groups
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item)[[intel::reqd_sub_group_size(8)]] {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# write items with vote functions
    all[i] = all_of_group(sg, input[i]);
    any[i] = any_of_group(sg, input[i]);
    none[i] = none_of_group(sg, input[i]);

  }).wait();

  std::cout << "all_of:\n";
  for(int i=0; i<N; i++) std::cout << all[i] << " "; std::cout << "\n";
  std::cout << "any_of:\n";
  for(int i=0; i<N; i++) std::cout << any[i] << " "; std::cout << "\n";
  std::cout << "none_of:\n";
  for(int i=0; i<N; i++) std::cout << none[i] << " "; std::cout << "\n";
  
  free(input, q);
  free(all, q);
  free(any, q);
  free(none, q);
  return 0;
}
