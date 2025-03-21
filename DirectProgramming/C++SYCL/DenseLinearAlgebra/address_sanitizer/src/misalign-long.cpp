#include <random>
#include <sycl/sycl.hpp>

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(1, 7);

  sycl::queue Q;
  constexpr std::size_t N = 4;
  auto *array = sycl::malloc_shared<long long>(N, Q);
  auto offset = distrib(gen);
  std::cout << "offset: " << offset << std::endl;
  array = (long long *)((char *)array + offset);  // BUG: root cause

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class MyKernel>(sycl::nd_range<1>(N, 1),
                                   [=](sycl::nd_item<1> item) { ++array[0]; }); // BUG: misalign access
    Q.wait();
  });

  return 0;
}
