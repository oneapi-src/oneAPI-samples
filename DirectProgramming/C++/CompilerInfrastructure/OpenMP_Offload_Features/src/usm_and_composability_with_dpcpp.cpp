//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <omp.h>
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>

extern "C" void *omp_target_get_context(int);

#ifdef OCL_BACKEND
#pragma omp requires unified_shared_memory
#endif

int main() {
  const unsigned Size = 200;
  int D = omp_get_default_device();
#ifdef OCL_BACKEND
  cl::sycl::queue Q(
      cl::sycl::context(static_cast<cl_context>(omp_target_get_context(D))),
      cl::sycl::gpu_selector());
#else
  cl::sycl::queue Q;
#endif
  std::cout << "SYCL: Running on "
            << Q.get_device().get_info<cl::sycl::info::device::name>() << "\n";
  if (!Q.get_device()
           .get_info<cl::sycl::info::device::usm_shared_allocations>()) {
    std::cout << "SYCL: USM is not available\n";
    return 0;
  }
  auto validate = [](int *Data) {
    for (unsigned I = 0; I < Size; ++I)
      if (Data[I] != 100 + I) return "failed";
    return "passed";
  };
  auto testOmp = [&](int *Data) {
    std::fill_n(Data, Size, -1);
#pragma omp target parallel for device(D)
    for (unsigned I = 0; I < Size; ++I) {
      Data[I] = 100 + I;
    }
    return validate(Data);
  };
  auto testDpc = [&](int *Data) {
    std::fill_n(Data, Size, -1);
    Q.submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for<class K>(cl::sycl::range<1>(Size),
                                [=](cl::sycl::id<1> I) { Data[I] = 100 + I; });
    });
    Q.wait();
    return validate(Data);
  };

  int *ompMem = (int *)omp_target_alloc_shared(Size * sizeof(int), D);
  int *dpcMem = cl::sycl::malloc_shared<int>(Size, Q);
  std::cout << "SYCL and OMP memory: " << testDpc(ompMem) << "\n";
  std::cout << "OMP and OMP memory:  " << testOmp(ompMem) << "\n";
  std::cout << "OMP and SYCL memory: " << testOmp(dpcMem) << "\n";
  std::cout << "SYCL and SYCL memory: " << testDpc(dpcMem) << "\n";
  omp_target_free(ompMem, D);
  cl::sycl::free(dpcMem, Q);
  return 0;
}
