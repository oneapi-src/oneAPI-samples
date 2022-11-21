//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include<oneapi/dpl/execution>
#include<oneapi/dpl/algorithm>
#include<oneapi/dpl/ranges>
#include<iostream>
#include<vector>

using namespace sycl;
using namespace oneapi::dpl::experimental::ranges;

int main()
{
    std::vector<int> v(20);

    {
        buffer A(v);
        auto view = iota_view(0, 20);
        auto rev_view = views::reverse(view);
        auto range_res = all_view<int, cl::sycl::access::mode::write>(A);

        copy(oneapi::dpl::execution::dpcpp_default, rev_view, range_res);
    }

    for (auto x : v)
        std::cout << x << " ";
    std::cout << "\n";
    return 0;
}
