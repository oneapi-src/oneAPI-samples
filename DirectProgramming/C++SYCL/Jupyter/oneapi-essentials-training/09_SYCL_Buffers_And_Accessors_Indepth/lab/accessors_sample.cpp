//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <sycl/sycl.hpp>
#include <cassert>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  // Create 3 buffers of 42 ints
  buffer<int> A{range{N}};
  buffer<int> B{range{N}};
  buffer<int> C{range{N}};  

  Q.submit([&](handler &h) {
      // create device accessors
      accessor aA{A, h, write_only, no_init};
      accessor aB{B, h, write_only, no_init};
      accessor aC{C, h, write_only, no_init};
      h.parallel_for(N, [=](id<1> i) {
          aA[i] = 1;
          aB[i] = 40;
          aC[i] = 0;
        });
    });
  Q.submit([&](handler &h) {
      // create device accessors
      accessor aA{A, h, read_only};
      accessor aB{B, h, read_only};
      accessor aC{C, h, read_write};
      h.parallel_for(N, [=](id<1> i) { aC[i] += aA[i] + aB[i]; });
    }); 

  host_accessor result{C, read_only};
    
  for (int i = 0; i < N; i++) std::cout << result[i] << " ";  
  
  return 0;
}
