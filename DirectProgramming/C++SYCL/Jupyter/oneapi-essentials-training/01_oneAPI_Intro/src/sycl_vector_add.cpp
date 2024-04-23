//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
//# STEP 1 : Include header for SYCL
#include <sycl/sycl.hpp>

int main(){
    
    //# STEP 2: Create a SYCL queue and device selection for offload
    sycl::queue q;

    //# initialize some data array
    const int N = 16;
    
    //# STEP 3: Allocate memory so that both host and device can access
    auto a = sycl::malloc_shared<float>(N, q);
    auto b = sycl::malloc_shared<float>(N, q);
    auto c = sycl::malloc_shared<float>(N, q);

    for(int i=0;i<N;i++) {
        a[i] = 1;
        b[i] = 2;
        c[i] = 0;
    }

    //# STEP 4: Submit computation to Offload device     
    q.parallel_for(N, [=](auto i){
        //# computation
        for(int i=0;i<N;i++) c[i] = a[i] + b[i];
    }).wait();

    //# print output
    for(int i=0;i<N;i++) std::cout << c[i] << "\n";
    
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
}
