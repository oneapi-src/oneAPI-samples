// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <CL/cl.h>
#include <CL/sycl/backend/opencl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  {
    buffer data_buf{data};

// BEGIN CODE SNIP
    // Note: This must select a device that supports interop with OpenCL kernel
    // objects!
    queue Q{ cpu_selector{} };
    context sc = Q.get_context();

    const char* kernelSource =
        R"CLC(
            kernel void add(global int* data) {
                int index = get_global_id(0);
                data[index] = data[index] + 1;
            }
        )CLC";
    cl_context c = get_native<backend::opencl>(sc);
    cl_program p =
        clCreateProgramWithSource(c, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(p, 0, nullptr, nullptr, nullptr, nullptr);
    cl_kernel k = clCreateKernel(p, "add", nullptr);

    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << "\n";

    kernel sk = make_kernel<backend::opencl>(k, sc);

    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};

      h.set_args(data_acc);
      h.parallel_for(size, sk);
    });

    clReleaseContext(c);
    clReleaseProgram(p);
    clReleaseKernel(k);
// END CODE SNIP
  }

  for (int i = 0; i < size; i++) {
    if (data[i] != i + 1) {
      std::cout << "Results did not validate at index " << i << "!\n";
      return -1;
    }
  }

  std::cout << "Success!\n";
  return 0;
}
