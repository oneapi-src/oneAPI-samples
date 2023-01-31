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
    //const int n = 10;
    //const int k = 5;

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
    std::vector<int> output_values(num_elements,0); 
 
  
    //Create buffers for the above vectors    

    buffer buf_in(input_seq);
    buffer buf_seq(input_pattern);    
    buffer buf_out(output_values);

    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto vals_end = oneapi::dpl::end(buf_seq);
    auto result_begin = oneapi::dpl::begin(buf_out);

    // use policy for algorithms execution
    auto policy = make_device_policy(q);  

    //function object to be passed to sort function  

    //Calling the onedpl binary search algorithm. We pass in the policy, the buffer iterators for the input vectors and the output. 
    // Default comparator is the operator < used here. 
    const auto i =  oneapi::dpl::binary_search(policy,keys_begin,keys_end,vals_begin,vals_end,result_begin);
   

    // 3.Checking results by creating the host accessors  
    host_accessor result_vals(buf_out,read_only);  

    std::cout<< "Input sequence = [";    
    std::copy(input_seq.begin(),input_seq.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search sequence = [";     
    std::copy(input_pattern.begin(),input_pattern.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Search results = [";    
    std::copy(output_values.begin(),output_values.end(),std::ostream_iterator<bool>(std::cout," "));
    std::cout <<"]"<< "\n";  
  
  return 0;
}
