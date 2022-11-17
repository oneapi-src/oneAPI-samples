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
  
    const int num_elements = 5;
    auto R = range(num_elements); 

    //Create queue with default selector  
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    //Initialize the input vector for search
    std::vector<int> input_seq{0, 2, 2, 2, 3, 3, 3, 3, 6, 6};
    //Initialize the input vector for search pattern
    std::vector<int> input_pattern{0, 2, 4, 7, 6};
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
    auto policy = make_device_policy(q);  

    //Calling the onedpl upper_bound algorithm. We pass in the policy, the buffer iterators for the input vectors and the output. 
    // Default comparator is the operator < used here.
    
    oneapi::dpl::lower_bound(policy,keys_begin,keys_end,vals_begin,vals_end,result_begin);   

    // 3.Checking results by creating the host accessors  
    host_accessor result_vals(buf_out,read_only);
    
    std::cout<< "Input Sequence = [ ";    
    std::copy(input_seq.begin(),input_seq.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search Sequence = [ ";     
    std::copy(input_pattern.begin(),input_pattern.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search Results = [ ";    
    std::copy(out_values.begin(),out_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}
