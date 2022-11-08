//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
using namespace sycl;

static const int N = 1024;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //intialize 2 arrays on host
  int *data1 = static_cast<int *>(malloc(N * sizeof(int)));
  int *data2 = static_cast<int *>(malloc(N * sizeof(int)));
  for (int i = 0; i < N; i++) {
    data1[i] = 25;
    data2[i] = 49;
  }
    
  //# STEP 1 : Create USM device allocation for data1 and data2

  //# YOUR CODE GOES HERE





    
  //# STEP 2 : Copy data1 and data2 to USM device allocation
    
  //# YOUR CODE GOES HERE  





  //# STEP 3 : Write kernel code to update data1 on device with sqrt of value

  q.parallel_for(N, [=](auto i) { 

    //# YOUR CODE GOES HERE 
      
  });



  //# STEP 3 : Write kernel code to update data2 on device with sqrt of value
    
  q.parallel_for(N, [=](auto i) { 

    //# YOUR CODE GOES HERE 
      
  });


  //# STEP 5 : Write kernel code to add data2 on device to data1
    
  q.parallel_for(N, [=](auto i) { 

  //# YOUR CODE GOES HERE 
      
  });



  //# STEP 6 : Copy data1 on device to host
    
  //# YOUR CODE GOES HERE 






  //# verify results
  int fail = 0;
  for (int i = 0; i < N; i++) if(data1[i] != 12) {fail = 1; break;}
  if(fail == 1) std::cout << " FAIL"; else std::cout << " PASS";
  std::cout << "\n";


  //# STEP 7 : Free USM device allocations

  //# YOUR CODE GOES HERE




  //# STEP 8 : Add event based kernel dependency for the Steps 2 - 6


  return 0;
}

