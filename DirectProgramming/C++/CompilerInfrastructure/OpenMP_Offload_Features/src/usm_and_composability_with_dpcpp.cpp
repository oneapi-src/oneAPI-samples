//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <omp.h>
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>

using namespace sycl;

extern "C" void *omp_target_get_context(int);

#ifdef OCL_BACKEND
#pragma omp requires unified_shared_memory
#endif

int main() {
  const unsigned kSize = 200;
  int d = omp_get_default_device();
#ifdef OCL_BACKEND
  sycl::queue q(
      sycl::context(static_cast<cl_context>(omp_target_get_context(d))),
      sycl::gpu_selector());
#else
  sycl::queue q;
#endif
  std::cout << "SYCL: Running on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  if (!q.get_device()
           .get_info<sycl::info::device::usm_shared_allocations>()) {
    std::cout << "SYCL: USM is not available\n";
    return 0;
  }
  auto Validate = [](int *data) {
    for (unsigned i = 0; i < kSize; ++i)
      if (data[i] != 100 + i) return "failed";
    return "passed";
  };
  auto TestOmp = [&](int *data) {
    std::fill_n(data, kSize, -1);
#pragma omp target parallel for device(d)
    for (unsigned i = 0; i < kSize; ++i) {
      data[i] = 100 + i;
    }
    return Validate(data);
  };
  auto TestDPCPP = [&](int *data) {
    std::fill_n(data, kSize, -1);
    q.parallel_for<class K>(sycl::range<1>(kSize), [=] (sycl::id<1> i)
		    {data[i] = 100 + i; }).wait();
    return Validate(data);
  };

  int *omp_mem = (int *)omp_target_alloc_shared(kSize * sizeof(int), d);
  int *dpcpp_mem = sycl::malloc_shared<int>(kSize, q);
  std::cout << "SYCL and OMP memory: " << TestDPCPP(omp_mem) << "\n";
  std::cout << "OMP and OMP memory:  " << TestOmp(omp_mem) << "\n";
  std::cout << "OMP and SYCL memory: " << TestOmp(dpcpp_mem) << "\n";
  std::cout << "SYCL and SYCL memory: " << TestDPCPP(dpcpp_mem) << "\n";
  omp_target_free(omp_mem, d);
  sycl::free(dpcpp_mem, q);
  return 0;
}
