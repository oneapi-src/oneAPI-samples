#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>

// Using oneDPL max_element with SYCL buffers.

int main() {

    std::vector<int> data{2, 2, 2, 4, 1, 1, 1};
    sycl::buffer<int> data_buf(data);

    auto policy = oneapi::dpl::execution::dpcpp_default;
    auto maxloc = oneapi::dpl::max_element(policy,
                                           oneapi::dpl::begin(data_buf),
                                           oneapi::dpl::end(data_buf));

    std::cout << "Run on "
              << policy.queue().get_device().template
                                        get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Maximum value is at element "
              << oneapi::dpl::distance(oneapi::dpl::begin(data_buf), maxloc)
              << std::endl;
    return 0;
}
