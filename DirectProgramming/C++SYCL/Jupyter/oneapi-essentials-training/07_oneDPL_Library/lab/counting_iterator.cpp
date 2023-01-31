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
    
    oneapi::dpl::counting_iterator<int> count_a(0);
    oneapi::dpl::counting_iterator<int> count_b = count_a + 100;
    int init = count_a[0]; // OK: init == 0
    //*count_b = 7; // ERROR: counting_iterator doesn't provide write operations
    auto sum = oneapi::dpl::reduce(dpl::execution::dpcpp_default,
     count_a, count_b, init); // sum is (0 + 0 + 1 + ... + 99) = 4950
    std::cout << "The Sum is: " <<sum<<"\n";
    
    return 0;
}
