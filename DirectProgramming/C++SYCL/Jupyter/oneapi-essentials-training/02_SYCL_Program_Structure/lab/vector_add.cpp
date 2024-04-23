//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
    const int N = 256;
    
    //# Initialize a vector and print values
    std::vector<int> vector1(N, 10);
    std::cout<<"\nInput Vector1: ";    
    for (int i = 0; i < N; i++) std::cout << vector1[i] << " ";

    //# STEP 1 : Create second vector, initialize to 20 and print values

    //# YOUR CODE GOES HERE
    
    
    
    
    //# Create Buffer
    
    buffer vector1_buffer(vector1);

    //# STEP 2 : Create buffer for second vector 

    //# YOUR CODE GOES HERE




    //# Submit task to add vector
    queue q;
    q.submit([&](handler &h) {
      //# Create accessor for vector1_buffer
      accessor vector1_accessor (vector1_buffer,h);
      
      //# STEP 3 - add second accessor for second buffer

      //# YOUR CODE GOES HERE



      h.parallel_for(range<1>(N), [=](id<1> index) {

        //# STEP 4 : Modify the code below to add the second vector to first one

        vector1_accessor[index] += 1;



      });
   });

 
  //# Create a host accessor to copy data from device to host
  host_accessor h_a(vector1_buffer,read_only);

  //# Print Output values 
  std::cout<<"\nOutput Values: ";
  for (int i = 0; i < N; i++) std::cout<< vector1[i] << " ";
  std::cout<<"\n";

  return 0;
}
