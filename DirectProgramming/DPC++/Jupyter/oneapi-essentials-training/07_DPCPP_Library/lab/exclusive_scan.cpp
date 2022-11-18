//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>


using namespace sycl;
using namespace oneapi::dpl::execution;



int main() {
    using T = int;
    const int num_elements = 6;    
    auto R = range(num_elements);     

    //Initialize the input vector for Keys
    std::vector<int> input_keys{ 0,0,0,1,1,1 };
    //Initialize the input vector for Values
    std::vector<int> input_values{ 1,2,3,4,5,6 };
    //Output vectors where we get the results back
    std::vector<int> output_values(num_elements, 0);

    //Create buffers for the above vectors    
    
    buffer buf_in(input_keys);
    buffer buf_seq(input_values);
    //buffer buf_out(output_values);
    buffer buf_out(output_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    //Create queue with default selector  
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    auto policy = make_device_policy(q);

    auto iter_res = oneapi::dpl::exclusive_scan_by_segment(policy, keys_begin, keys_end, vals_begin, result_begin,T(0));
    auto count_res = std::distance(result_begin,iter_res);    

    // 3.Checking results    
    host_accessor result_vals(buf_out,read_only);

    std::cout<< "Keys = [ ";    
    std::copy(input_keys.begin(),input_keys.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Values = [ ";     
    std::copy(input_values.begin(),input_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Output Values = [ ";    
    std::copy(output_values.begin(),output_values.begin() + count_res,std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}
