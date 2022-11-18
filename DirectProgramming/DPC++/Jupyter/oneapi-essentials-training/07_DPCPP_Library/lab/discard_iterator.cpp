//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>

#include <tuple>


using namespace sycl;
using namespace oneapi::dpl::execution;
using std::get;


int main() {

    const int num_elements = 10;

    //Initialize the input vector for search
    std::vector<int> input_seq{2, 4, 12, 24, 34, 48, 143, 63, 76, 69};
    //Initialize the stencil values
    std::vector<int> input_pattern{1, 2, 4, 1, 6, 1, 2, 1, 7, 1};
    //Output vector where we get the results back
    std::vector<int> out_values(num_elements,0);


    buffer buf_in(input_seq);
    buffer buf_seq(input_pattern);
    buffer buf_out(out_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto vals_end = oneapi::dpl::end(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    auto policy = oneapi::dpl::execution::dpcpp_default;

    auto zipped_first = oneapi::dpl::make_zip_iterator(keys_begin, vals_begin);

    auto iter_res = oneapi::dpl::copy_if(dpl::execution::dpcpp_default,zipped_first, zipped_first + num_elements,
                 dpl::make_zip_iterator(result_begin, dpl::discard_iterator()),
                 [](auto t){return get<1>(t) == 1;});    
    

    // 3.Checking results by creating the host accessors  
    host_accessor result_vals(buf_out,read_only);

    std::cout<< "Input Sequence = [ ";    
    std::copy(input_seq.begin(),input_seq.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Sequence to search = [ ";     
    std::copy(input_pattern.begin(),input_pattern.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Results with stencil value of 1 = [ ";    
    std::copy(out_values.begin(),out_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}
