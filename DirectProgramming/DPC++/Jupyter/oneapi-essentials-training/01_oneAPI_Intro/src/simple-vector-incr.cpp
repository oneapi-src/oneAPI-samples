//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;
//N is set as 2 as this is just for demonstration purposes. Even if you make N bigger than 2 the program still
//counts N as only 2 as the first 2 elements are only initialized here and the rest all becomes zero.
static const size_t N = 2;

// ############################################################
// work

void work(queue &q) {
  std::cout << "Device : "
            << q.get_device().get_info<info::device::name>()
            << "\n";
  // ### Step 1 - Inspect
  // The code presents one input buffer (vector1) for which Sycl buffer memory
  // is allocated. The associated with vector1_accessor set to read/write gets
  // the contents of the buffer.
  int vector1[N] = {10, 10};
  auto R = range(N);
  
  std::cout << "Input  : " << vector1[0] << ", " << vector1[1] << "\n";

  // ### Step 2 - Add another input vector - vector2
  // Uncomment the following line to add input vector2
  //int vector2[N] = {20, 20};

  // ### Step 3 - Print out for vector2
  // Uncomment the following line
  //std::cout << "Input  : " << vector2[0] << ", " << vector2[1] << "\n";
  buffer vector1_buffer(vector1,R);

  // ### Step 4 - Add another Sycl buffer - vector2_buffer
  // Uncomment the following line
  //buffer vector2_buffer(vector2,R);
  q.submit([&](handler &h) {
    accessor vector1_accessor (vector1_buffer,h);

    // Step 5 - add an accessor for vector2_buffer
    // Uncomment the following line to add an accessor for vector 2
    //accessor vector2_accessor (vector2_buffer,h,read_only);

    h.parallel_for<class test>(range<1>(N), [=](id<1> index) {
      // ### Step 6 - Replace the existing vector1_accessor to accumulate
      // vector2_accessor 
      // Comment the following line
      vector1_accessor[index] += 1;

      // Uncomment the following line
      //vector1_accessor[index] += vector2_accessor[index];
    });
  });
  q.wait();
  host_accessor h_a(vector1_buffer,read_only);
  std::cout << "Output : " << vector1[0] << ", " << vector1[1] << "\n";
}

// ############################################################
// entry point for the program

int main() {  
  try {
    queue q;
    work(q);
  } catch (exception e) {
    std::cerr << "Exception: " << e.what() << "\n";
    std::terminate();
  } catch (...) {
    std::cerr << "Unknown exception" << "\n";
    std::terminate();
  }
}
