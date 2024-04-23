//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

using namespace sycl;

// kernel function to compute vector add using Unified Shared memory model (USM)
void kernel_usm(int* a, int* b, int* c, int N) {
  //Step 1: create a device queue
  queue q;
  //Step 2: create USM device allocation
  auto a_device = malloc_device<int>(N, q); 
  auto b_device = malloc_device<int>(N, q); 
  auto c_device = malloc_device<int>(N, q); 
  //Step 3: copy memory from host to device
  q.memcpy(a_device, a, N*sizeof(int));
  q.memcpy(b_device, b, N*sizeof(int));
  q.wait();
  //Step 4: send a kernel (lambda) for execution
  q.parallel_for(N, [=](auto i){
    //Step 5: write a kernel
    c_device[i] = a_device[i] + b_device[i];
  }).wait();
  //Step 6: copy the result back to host
  q.memcpy(c, c_device, N*sizeof(int)).wait();
  //Step 7: free device allocation
  free(a_device, q);
  free(b_device, q);
  free(c_device, q);
}

// kernel function to compute vector add using Buffer memory model
void kernel_buffers(int* a, int* b, int* c, int N) {
  //Step 1: create a device queue
  queue q;
  //Step 2: create buffers 
  buffer buf_a(a, range<1>(N));
  buffer buf_b(b, range<1>(N));
  buffer buf_c(c, range<1>(N));
  //Step 3: submit a command for (asynchronous) execution
  q.submit([&](handler &h){
    //Step 4: create buffer accessors to access buffer data on the device
    accessor A(buf_a, h, read_only);
    accessor B(buf_b, h, read_only);
    accessor C(buf_c, h, write_only);
    //Step 5: send a kernel (lambda) for execution
    h.parallel_for(N, [=](auto i){
      //Step 6: write a kernel
      C[i] = A[i] + B[i];
    });
  });
}

int main() {
  // initialize data arrays on host
  constexpr int N = 256;
  int a[N], b[N], c[N];
  for (int i=0; i<N; i++){
    a[i] = 1;
    b[i] = 2;
  }
    
  // initialize c = 0 and offload computation using USM, print output 
  for (int i=0; i<N; i++) c[i] = 0;
  kernel_usm(a, b, c, N);
  std::cout << "Vector Add Output (USM): \n";
  for (int i=0; i<N; i++)std::cout << c[i] << " ";std::cout << "\n";

  // initialize c = 0 and offload computation using USM, print output 
  for (int i=0; i<N; i++) c[i] = 0;
  std::cout << "Vector Add Output (Buffers): \n";
  kernel_buffers(a, b, c, N);
  for (int i=0; i<N; i++)std::cout << c[i] << " ";std::cout << "\n";
    
}
