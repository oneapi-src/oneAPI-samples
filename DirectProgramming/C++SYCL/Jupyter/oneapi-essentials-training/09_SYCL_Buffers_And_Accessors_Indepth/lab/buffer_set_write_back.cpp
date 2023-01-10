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
    
  {
    queue q;
    buffer my_buffer(my_data);
      
    //Call the set_write_back method to control the data to be written back to the host from the device. e
    //Setting it to false will not update the host with the updated values
         
    my_buffer.set_write_back(false);    

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
  
}
