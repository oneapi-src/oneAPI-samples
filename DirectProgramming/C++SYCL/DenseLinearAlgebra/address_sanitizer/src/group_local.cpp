#include <sycl/sycl.hpp>

constexpr std::size_t N = 16;
constexpr std::size_t group_size = 8;

int main() {
  sycl::queue Q;
  auto *data = sycl::malloc_host<int>(1, Q);

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(
        sycl::nd_range<1>(N, group_size), [=](sycl::nd_item<1> item) {
          auto &ref = *sycl::ext::oneapi::group_local_memory<int[N]>(
                  item.get_group());
          ref[item.get_local_linear_id() * 2 + 4] = 42; // BUG: out-of-bounds on local memory
        });
  });

  Q.wait();
  return 0;
}
