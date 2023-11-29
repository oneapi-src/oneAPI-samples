//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include <iostream>

int main(){
    //# select device for offload
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    //# initialize some data array
    const int N = 16;
    auto data = sycl::malloc_shared<float>(N, q);
    for(int i=0;i<N;i++) data[i] = i;

    //# parallel computation on GPU
    q.parallel_for(N,[=](auto i){
        data[i] = data[i] * 5;
    }).wait();

    //# print output
    for(int i=0;i<N;i++) std::cout << data[i] << "\n"; 
}
