//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <stdlib.h>
#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 1024;

void dpcpp_code(int *a, int *b, int *c) {
  queue q{default_selector()};  // Create Command Queue Targeting GPU
  std::cout << "Device: " << q.get_device().get_info<info::device::name>()
            << std::endl;

  program p(q.get_context());  // Create program from the same context as q

  // Compile OpenCL vecAdd kernel. which is expressed as a C++ Raw String
  // as indicated by R”
  p.build_with_source(R"( __kernel void vecAdd(__global int *a, 
                                                __global int *b, 
                                                __global int *c) 
                            {
                                int i=get_global_id(0);
                                c[i] = a[i] + b[i]; 
                            } )");

  buffer<int, 1> buf_a(a, range<1>(N));
  buffer<int, 1> buf_b(b, range<1>(N));
  buffer<int, 1> buf_c(c, range<1>(N));

  q.submit([&](handler &h) {
    auto A = buf_a.get_access<access::mode::read>(h);
    auto B = buf_b.get_access<access::mode::read>(h);
    auto C = buf_c.get_access<access::mode::write>(h);
    // Set buffers as arguments to the kernel
    h.set_args(A, B, C);
    // Launch vecAdd kernel from the p program object across N elements.
    h.parallel_for(range<1>(N), p.get_kernel("vecAdd"));
  });
}

int main(int argc, char **argv) {
  // Ensure to use OpenCL backend for OpenCL Kernel Compilation
  putenv((char *)"SYCL_DEVICE_FILTER=OPENCL");

  size_t bytes = sizeof(int) * N;

  int *a = (int *)malloc(bytes);
  int *b = (int *)malloc(bytes);
  int *c = (int *)malloc(bytes);
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }
  dpcpp_code(a, b, c);

  for (int i = 0; i < N; ++i) {
    if (c[i] != i * 3) {
      std::cout << "FAILED!" << std::endl;
      return -1;
    }
  }
  std::cout << "PASSED!" << std::endl;

  free(a);
  free(b);
  free(c);
  return 0;
}