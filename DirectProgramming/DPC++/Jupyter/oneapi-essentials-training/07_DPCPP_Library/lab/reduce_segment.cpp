
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <CL/sycl.hpp>

using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
        
    const int num_elements = 6;    
    auto R = range(num_elements);    
    
    //Create queue with default selector  
    queue q;
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    //Initialize the input vector for Keys
    std::vector<int> input_keys{ 0,0,0,1,1,1 };
    //Initialize the input vector for Values
    std::vector<int> input_values{ 1,2,3,4,5,6 };
    //Output vectors where we get the results back
    std::vector<int> output_keys(num_elements, 0);
    std::vector<int> output_values(num_elements, 0);    
    
    //Create buffers for the above vectors    
    buffer buf_in(input_keys);
    buffer buf_seq(input_values);    
    buffer buf_out_keys(output_keys.data(),R);
    buffer buf_out_vals(output_values.data(),R);


    // create buffer iterators
    auto keys_begin = oneapi::dpl::begin(buf_in);
    auto keys_end = oneapi::dpl::end(buf_in);
    auto vals_begin = oneapi::dpl::begin(buf_seq);
    auto result_key_begin = oneapi::dpl::begin(buf_out_keys);
    auto result_vals_begin = oneapi::dpl::begin(buf_out_vals);

    // use policy for algorithms execution
    auto policy = make_device_policy(q);
    //auto pair_iters = make_pair <std::vector::iterator, std::vector::iterator>

    //Calling the oneDPL reduce by search algorithm. We pass in the policy, the buffer iterators for the input vectors and the output. 
    // Default comparator is the operator < used here.
    // dpl::reduce_by_segment returns a pair of iterators to the result_key_begin and result_vals_begin respectively
    int count_keys,count_vals = 0;    
    
    auto pair_iters = oneapi::dpl::reduce_by_segment(make_device_policy(q), keys_begin, keys_end, vals_begin, result_key_begin, result_vals_begin);
    auto iter_keys = std::get<0>(pair_iters);    
    // get the count of the items in the result_keys using std::distance
    count_keys = std::distance(result_key_begin,iter_keys);    
    //get the second iterator
    auto iter_vals = std::get<1>(pair_iters);    
    count_vals = std::distance(result_vals_begin,iter_vals);    

    // 3.Checking results by creating the host accessors    
    host_accessor result_keys(buf_out_keys,read_only);
    host_accessor result_vals(buf_out_vals,read_only); 
    

    std::cout<< "Keys = [ ";    
    std::copy(input_keys.begin(),input_keys.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Values = [ ";     
    std::copy(input_values.begin(),input_values.end(),std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    std::cout<< "Output Keys = [ ";    
    std::copy(output_keys.begin(),output_keys.begin() + count_keys,std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";
    
    std::cout<< "Output Values = [ ";    
    std::copy(output_values.begin(),output_values.begin() + count_vals,std::ostream_iterator<int>(std::cout," "));
    std::cout <<"]"<< "\n";

    return 0;
}
