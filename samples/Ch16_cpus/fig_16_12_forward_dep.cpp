// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

int main()
{
  const int n = 16, w = 8;

  queue Q;
  range<2> G = {n, w};
  range<2> L = {1, w};

  int *a = malloc_shared<int>(n*(n+1), Q);

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n+1; j++) a[i*n + j] = i + j;

  Q.parallel_for(nd_range<2>{G, L}, [=](nd_item<2> it)
    [[intel::reqd_sub_group_size(w)]] {

    // distribute uniform "i" over the sub-group with 8-way
    // redundant computation
    const int i = it.get_global_id(0);
    sub_group sg = it.get_sub_group();

    for (int j = sg.get_local_id()[0]; j < n; j += w) {
      // load a[i*n+j+1:8] before updating a[i*n+j:8] to preserve
      // loop-carried forward dependency
      auto va = a[i*n + j + 1];
      sg.barrier();
      a[i*n + j] = va + i + 2;
    }
    sg.barrier();

  }).wait();

  if (a[0] == 3 && a[9] == 12)
    std::cout << "passed\n";
  else
    std::cout << "failed\n";
  free(a, Q);
}
