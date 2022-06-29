#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iostream>

// Using oneDPL max_element with USM.

int main() {

    sycl::queue Q(sycl::default_selector{});
    auto policy = oneapi::dpl::execution::make_device_policy(Q);

    const size_t n = 7;
    auto data = sycl::malloc_shared<int>(n, Q);

    data[0] = 2;
    data[1] = 2;
    data[2] = 2;
    data[3] = 4;
    data[4] = 1;
    data[5] = 1;
    data[6] = 1;

    auto maxloc = oneapi::dpl::max_element(policy, data, data + n);
    policy.queue().wait();

    std::cout << "Run on "
              << policy.queue().get_device().template
                                        get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Maximum value is at element "
              << oneapi::dpl::distance(data, maxloc)
              << std::endl;

    sycl::free(data, Q);
    return 0;
}
