//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    constexpr int num_elements = 16;
    std::vector<int> input_v1(num_elements, 2), input_v2(num_elements, 5), input_v3(num_elements, 0);
    //Zip Iterator zips up the iterators of individual containers of interest.
    auto start = oneapi::dpl::make_zip_iterator(input_v1.begin(), input_v2.begin(), input_v3.begin());
    auto end = oneapi::dpl::make_zip_iterator(input_v1.end(), input_v2.end(), input_v3.end());
    //create device policy
    auto exec_policy = make_device_policy(q);
    oneapi::dpl::for_each(exec_policy, start, end, [](auto t) {
        //The zip iterator is used for expressing bounds in PSTL algorithms.
        using std::get;
        get<2>(t) = get<1>(t) * get<0>(t);
        });
    for (auto it = input_v3.begin(); it < input_v3.end(); it++)
    std::cout << (*it) <<" ";
    std::cout << "\n";
    return 0;
}
