//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>

using namespace sycl;
using namespace std;

struct multiply_index_by_two {
    template <typename Index>
    Index operator[](const Index& i) const
    {
        return i * 2;
    }
};

int main() {
    //queue q;
    const int num_elelemts = 100;
    std::vector<float> result(num_elelemts, 0);
    oneapi::dpl::counting_iterator<int> first(0);
    oneapi::dpl::counting_iterator<int> last(20);

    // first and last are iterators that define a contiguous range of input elements
    // compute the number of elements in the range between the first and last that are accessed
    // by the permutation iterator
    size_t num_elements = std::distance(first, last) / 2 + std::distance(first, last) % 2;
    using namespace oneapi;
    auto permutation_first = oneapi::dpl::make_permutation_iterator(first, multiply_index_by_two());
    auto permutation_last = permutation_first + num_elements;
    auto it = ::std::copy(oneapi::dpl::execution::dpcpp_default, permutation_first, permutation_last, result.begin());
    auto count = ::std::distance(result.begin(),it);
    
    for(int i = 0; i < count; i++) ::std::cout << result[i] << " ";
    
   // for (auto it = result.begin(); it < result.end(); it++)     
    //   ::std::cout << (*it) <<" "; 
        
    return 0;
}
