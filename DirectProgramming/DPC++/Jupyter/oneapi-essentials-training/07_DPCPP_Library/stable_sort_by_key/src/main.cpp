//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>
#include <sycl/sycl.hpp>


using namespace sycl;
using namespace oneapi::dpl::execution;


using namespace std;

int main() {
  const int n = 1000000;
  buffer<int> keys_buf{n};  // buffer with keys
  buffer<int> vals_buf{n};  // buffer with values

  // create objects to iterate over buffers
  auto keys_begin = oneapi::dpl::begin(keys_buf);
  auto vals_begin = oneapi::dpl::begin(vals_buf);

  auto counting_begin = oneapi::dpl::counting_iterator<int>{0};
  // use default policy for algorithms execution
  auto policy = oneapi::dpl::execution::dpcpp_default;

  // 1. Initialization of buffers
  // let keys_buf contain {n, n, n-2, n-2, ..., 4, 4, 2, 2}
  transform(policy, counting_begin, counting_begin + n, keys_begin,
            [n](int i) { return n - (i / 2) * 2; });
  // fill vals_buf with the analogue of std::iota using counting_iterator
  copy(policy, counting_begin, counting_begin + n, vals_begin);

  // 2. Sorting
  auto zipped_begin = oneapi::dpl::make_zip_iterator(keys_begin, vals_begin);
  // stable sort by keys
  stable_sort(
      policy, zipped_begin, zipped_begin + n,
      // Generic lambda is needed because type of lhs and rhs is unspecified.
      [](auto lhs, auto rhs) { return get<0>(lhs) < get<0>(rhs); });

  // 3.Checking results
  //host_accessor host_keys(keys_buf,read_only);
  //host_accessor host_vals(vals_buf,read_only);
  auto host_keys = keys_buf.get_access<access::mode::read>();
  auto host_vals = vals_buf.get_access<access::mode::read>();

  // expected output:
  // keys: {2, 2, 4, 4, ..., n - 2, n - 2, n, n}
  // vals: {n - 2, n - 1, n - 4, n - 3, ..., 2, 3, 0, 1}
  for (int i = 0; i < n; ++i) {
    if (host_keys[i] != (i / 2) * 2 &&
        host_vals[i] != n - (i / 2) * 2 - (i % 2 == 0 ? 2 : 1)) {
      cout << "fail: i = " << i << ", host_keys[i] = " << host_keys[i]
           << ", host_vals[i] = " << host_vals[i] << "\n";
      return 1;
    }
  }

  cout << "success\nRun on "
       << policy.queue().get_device().template get_info<info::device::name>()
       << "\n";
  return 0;
}

