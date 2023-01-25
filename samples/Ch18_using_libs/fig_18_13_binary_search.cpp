// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// -------------------------------------------------------
// Changed from Book:
// dpstd:: is now
//   oneapi::dpl::
// dpstd::execution::default_policy is now
//   oneapi::dpl::execution::dpcpp_default
// -------------------------------------------------------

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main()
{
    buffer<uint64_t, 1> kB{ range<1>(10) };
    buffer<uint64_t, 1> vB{ range<1>(5) };
    buffer<uint64_t, 1> rB{ range<1>(5) };
    {
      host_accessor k{kB};
      host_accessor v{vB};

      // Initialize data, sorted
      k[0] = 0; k[1] = 5; k[2] = 6; k[3] = 6; k[4] = 7;
      k[5] = 7; k[6] = 8; k[7] = 8; k[8] = 9; k[9] = 9;

      v[0] = 1; v[1] = 6; v[2] = 3; v[3] = 7; v[4] = 8;
    }

    // create dpc++ iterators
    auto k_beg = oneapi::dpl::begin(kB);
    auto k_end = oneapi::dpl::end(kB);
    auto v_beg = oneapi::dpl::begin(vB);
    auto v_end = oneapi::dpl::end(vB);
    auto r_beg = oneapi::dpl::begin(rB);

    // create named policy from existing one
    auto policy = oneapi::dpl::execution::make_device_policy<class bSearch>(oneapi::dpl::execution::dpcpp_default);

    // call algorithm
    oneapi::dpl::binary_search(policy, k_beg, k_end, v_beg, v_end, r_beg);

    // check data
    host_accessor r{rB};
    if ((r[0] == false) && (r[1] == true) && (r[2] == false) && (r[3] == true) && (r[4] == true)) {
       std::cout << "Passed. \nRun on "
            << policy.queue().get_device().get_info<info::device::name>() << "\n";
    }
    else
       std::cout << "failed: values do not match.\n";

    return 0;
}
