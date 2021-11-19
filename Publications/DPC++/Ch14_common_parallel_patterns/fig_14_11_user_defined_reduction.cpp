// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// -------------------------------------------------------
// Changed from Book:
//   dropped 'using namespace sycl::ONEAPI'
//   this allows reduction to use the sycl::reduction,
//   added sycl::ONEAPI:: to minimum.
// -------------------------------------------------------

#include <CL/sycl.hpp>
#include <iostream>
#include <random>

using namespace sycl;

template <typename T, typename I>
struct pair {
  bool operator<(const pair& o) const {
    return val <= o.val || (val == o.val && idx <= o.idx);
  }
  T val;
  I idx;
};

template <typename T, typename I>
using minloc = minimum<pair<T, I>>;

int main() {
  constexpr size_t N = 16;
  constexpr size_t L = 4;

  queue Q;
  float* data = malloc_shared<float>(N, Q);
  pair<float, int>* res = malloc_shared<pair<float, int>>(1, Q);
  std::generate(data, data + N, std::mt19937{});

  pair<float, int> identity = {
      std::numeric_limits<float>::max(), std::numeric_limits<int>::min()};
  *res = identity;

  auto red = sycl::reduction(res, identity, minloc<float, int>());

  Q.submit([&](handler& h) {
     h.parallel_for(nd_range<1>{N, L}, red, [=](nd_item<1> item, auto& res) {
       int i = item.get_global_id(0);
       pair<float, int> partial = {data[i], i};
       res.combine(partial);
     });
   }).wait();

  std::cout << "minimum value = " << res->val << " at " << res->idx << "\n";

  pair<float, int> gold = identity;
  for (int i = 0; i < N; ++i) {
    if (data[i] <= gold.val || (data[i] == gold.val && i < gold.idx)) {
      gold.val = data[i];
      gold.idx = i;
    }
  }
  bool passed = (res->val == gold.val) && (res->idx == gold.idx);
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(res, Q);
  free(data, Q);
  return (passed) ? 0 : 1;
}
