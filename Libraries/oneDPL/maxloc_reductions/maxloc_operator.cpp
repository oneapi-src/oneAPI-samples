#include <iostream>
#include <CL/sycl.hpp>

// Hand-coded SYCL maxloc reduction operator.

template <typename T, typename I>
struct pair {
    bool operator > (const pair& o) const {
        return val >= o.val || (val == o.val && idx >= o.idx);
    }
    T val;
    I idx;
};

template <typename T, typename I>
using maxloc = sycl::maximum<pair<T, I>>;

const size_t L = 1;

int main(int argc, char **argv)
{
    sycl::queue Q(sycl::default_selector{});

    const size_t n = 7;
    auto data = sycl::malloc_shared<int>(n, Q);

    data[0] = 2;
    data[1] = 2;
    data[2] = 2;
    data[3] = 4;
    data[4] = 1;
    data[5] = 1;
    data[6] = 1;

    pair<int, int>* max_res = sycl::malloc_shared<pair<int, int>>(1, Q);
    pair<int, int> max_identity = {
        std::numeric_limits<int>::min(), std::numeric_limits<int>::min()
    };
    *max_res = max_identity;
    auto red_max = sycl::reduction(max_res, max_identity, maxloc<int, int>());

    Q.submit([&](sycl::handler& h)
    {
        h.parallel_for(sycl::nd_range<1>{n, L},
                       red_max,
                       [=](sycl::nd_item<1> item, auto& max_res)
        {
            int i = item.get_global_id(0);
            pair<int, int> partial = {data[i], i};
            max_res.combine(partial);
        });
    }).wait();

    std::cout << "Run on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    std::cout << "Maximum value = "
              << max_res->val
              << " at location " << max_res->idx
              << std::endl;

    //Cleanup
    sycl::free(data, Q);
    sycl::free(max_res, Q);

    return 0;
}
