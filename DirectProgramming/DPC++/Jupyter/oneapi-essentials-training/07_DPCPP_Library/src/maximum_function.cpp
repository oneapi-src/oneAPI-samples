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
    
    queue q;
    constexpr int N = 8;
    
    std::vector<int> v{-3,1,4,-1,5,9,-2,6}; 
    //create a separate scope for buffer destruction
    std::vector<int>result(N);
    {
        buffer<int,1> buf(v.data(), range<1>(N));
        buffer<int,1> buf_res(result.data(), range<1>(N));
        
        //dpstd buffer iterators for both the input and the result vectors
        auto start_v = oneapi::dpl::begin(buf);
        auto end_v = oneapi::dpl::end(buf);
        auto start_res = oneapi::dpl::begin(buf_res);
        auto end_res = oneapi::dpl::end(buf_res);
        
        //use std::fill to initialize the result vector
        std::fill(oneapi::dpl::execution::dpcpp_default,start_res, end_res, 0);  
        //usage of dpstd::maximum<> function call within the std::exclusive_scan function
        std::exclusive_scan(oneapi::dpl::execution::dpcpp_default, start_v, end_v, start_res, int(0), oneapi::dpl::maximum<int>() );        
    }
    
    
    for(int i = 0; i < result.size(); i++) std::cout << result[i] << "\n";
    return 0;
}
