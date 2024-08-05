//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <iostream>

int main(){
    //# initialize some data array
    const int N = 16;
    std::vector<int> data(N);
    for(int i=0;i<N;i++) data[i] = i;

    //# parallel computation on GPU using SYCL library (oneDPL)
    oneapi::dpl::for_each(oneapi::dpl::execution::dpcpp_default, data.begin(), data.end(), [](int &tmp){ tmp *= 5; });

    //# print output
    for(int i=0;i<N;i++) std::cout << data[i] << "\n"; 
}
