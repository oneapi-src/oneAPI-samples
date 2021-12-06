//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 256;

int main() {
  /* in_order queue property */
  queue q{property::queue::in_order()};
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  int *data = static_cast<int *>(malloc_shared(N * sizeof(int), q));
  for (int i = 0; i < N; i++) data[i] = 10;

  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });

  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });

  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 5; });
  q.wait();

  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n";
  free(data, q);
  return 0;
}
