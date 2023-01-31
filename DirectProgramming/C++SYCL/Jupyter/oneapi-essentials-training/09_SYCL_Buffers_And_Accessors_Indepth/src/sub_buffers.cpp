//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
    const int N = 64;
    const int num1 = 2;
    const int num2 = 3;
    int data[N];
    for (int i = 0; i < N; i++) data[i] = i;
    
    std::cout<<"BUffer Values: ";    
    for (int i = 0; i < N; i++) std::cout << data[i] << " "; 
    std::cout<<"\n";
    buffer B(data, range(N));

    //Create sub buffers with offsets and half of the range. 

    buffer<int> B1(B, 0, range{ N / 2 });
    buffer<int> B2(B, 32, range{ N / 2 });

    //Multiply the  elemets in first sub buffer by 2 
    queue q1;
    q1.submit([&](handler& h) {
        accessor a1(B1, h);
        h.parallel_for(N/2, [=](auto i) { a1[i] *= num1; });
    });

    //Multiply the  elemets in second sub buffer by 3    
    queue q2;
    q2.submit([&](handler& h) {
        accessor a2(B2, h);
        h.parallel_for(N/2, [=](auto i) { a2[i] *= num2; });
    });    
    
    //Host accessors to get the results back to the host from the device
    host_accessor b1(B1, read_only);
    host_accessor b2(B2, read_only);
    
    std::cout<<"Sub Buffer1: ";
    for (int i = 0; i < N/2; i++) std::cout<< b1[i] << " ";
    std::cout<<"\n";
    std::cout<<"Sub Buffer2: ";
    for (int i = 0; i < N/2; i++) std::cout << b2[i] << " ";

    return 0;
}
