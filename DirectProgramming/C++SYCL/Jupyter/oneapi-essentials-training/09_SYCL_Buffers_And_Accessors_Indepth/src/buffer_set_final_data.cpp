//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <array>
using namespace sycl;
constexpr int N = 42;

int main() {
  std::array<int,N> my_data;        
  for (int i = 0; i < N; i++)
        my_data[i] = i;     
 
  auto buff = std::make_shared<std::array<int, N>>(); 
  
  {
    queue q;
    buffer my_buffer(my_data);
      
    //Call the set_final_data to the created shared ptr where the values will be written back when the buffer gets destructed.
    //my_buffer.set_final_data(nullptr);    
    my_buffer.set_final_data(buff);   

    q.submit([&](handler &h) {
        // create an accessor to update
        // the buffer on the device
        accessor my_accessor(my_buffer, h);

        h.parallel_for(N, [=](id<1> i) {
            my_accessor[i]*=2;
          });
      });    
  }

  // myData is updated when myBuffer is
  // destroyed upon exiting scope
 
  for (int i = 0; i < N; i++) {
    std::cout << my_data[i] << " ";
  }
  std::cout << "\n"; 
  for (int i = 0; i < N; i++) {
    std::cout <<(*buff)[i] << " ";
  }
  std::cout << "\n"; 
  
}
