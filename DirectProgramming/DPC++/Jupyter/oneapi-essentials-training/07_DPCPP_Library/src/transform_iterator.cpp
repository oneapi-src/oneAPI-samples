//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/numeric>

using namespace sycl;
using namespace oneapi::dpl::execution;
        
int main() {
    
    dpl::counting_iterator<int> first(0);
    dpl::counting_iterator<int> last(100);
    auto func = [](const auto &x){ return x * 2; };
    auto transform_first = dpl::make_transform_iterator(first, func);
    auto transform_last = transform_first + (last - first);
    auto sum = std::reduce(dpl::execution::dpcpp_default,
         transform_first, transform_last); // sum is (0 + -1 + ... + -9) = -45   
    std::cout <<"Reduce output using Transform Iterator: "<<sum << "\n";
    return 0;
}
