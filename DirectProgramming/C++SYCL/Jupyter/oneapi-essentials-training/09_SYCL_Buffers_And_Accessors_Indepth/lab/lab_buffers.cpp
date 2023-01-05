//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
    const int N = 256;
    int data[N];
    for (int i = 0; i < N; i++) data[i] = i;
    
    std::cout<<"\nInput Values: ";    
    for (int i = 0; i < N; i++) std::cout << data[i] << " "; 
    std::cout<<"\n";
    buffer buf_data(data, range(N));

    //# STEP 1 : Create 3 sub-buffers for buf_data with length 64, 128 and 64. 

    //# YOUR CODE GOES HERE




    //# STEP 2 : Submit task to Multiply the  elements in first sub buffer by 2 
    queue q1;
    q1.submit([&](handler& h) {

      //# YOUR CODE GOES HERE



    });

    //# STEP 3 : Submit task to Multiply the  elements in second sub buffer by 3    
    queue q2;
    q2.submit([&](handler& h) {

      //# YOUR CODE GOES HERE



    });    

    //# STEP 4 : Submit task to Multiply the  elements in third sub buffer by 2    
    queue q3;
    q3.submit([&](handler& h) {

      //# YOUR CODE GOES HERE



    });  

    //# STEP 5 : Create Host accessors to get the results back to the host from the device

    //# YOUR CODE GOES HERE



    
    std::cout<<"\nOutput Values: ";
    for (int i = 0; i < N; i++) std::cout<< data[i] << " ";
    std::cout<<"\n";

    return 0;
}
