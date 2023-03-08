// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;


int main() {
  constexpr int global_size = 16;
  constexpr int local_size = 16;
  buffer<int,2> B{ range{ global_size, global_size }};

  queue gpu_Q{ gpu_selector_v };
  queue cpu_Q{ cpu_selector_v };

  nd_range NDR {
    range{ global_size, global_size },
    range{ local_size, local_size }};

  gpu_Q.submit([&](handler& h){
    accessor acc{B, h};

      h.parallel_for( NDR , [=](auto id) {
          auto ind = id.get_global_id();
          acc[ind] = ind[0] + ind[1];
          });
      }, cpu_Q); /** <<== Fallback Queue Specified **/

  host_accessor acc{B};
  for(int i=0; i < global_size; i++){
    for(int j = 0; j < global_size; j++){
      if( acc[i][j] != i+j ) {
        std::cout<<"Wrong result\n";
        return 1;
  } } }
  std::cout<<"Correct results\n";
  return 0;
}

