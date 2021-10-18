//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <mutex>
#include <CL/sycl.hpp>
using namespace sycl;
static const int N = 20;

int main() {
int myInts[N];
queue q;
//Initialize vector a,b and c
std::vector<float> a(N, 10.0f);
std::vector<float> b(N, 20.0f);

auto R = range<1>(N);
{
    //Create host_ptr buffers for a and b
    buffer buf_a(a,{property::buffer::use_host_ptr()});
    buffer buf_b(b,{property::buffer::use_host_ptr()});    
    
    q.submit([&](handler& h) {
        //create Accessors for a and b
        accessor A(buf_a,h);
        accessor B(buf_b,h,read_only);        
        h.parallel_for(R, [=](auto i) { A[i] += B[1] ; });
      });
}
    
for (int i = 0; i < N; i++) std::cout << a[i] << " ";
return 0;
}
