#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

sycl::ext::oneapi::experimental::device_global<
    int[4], decltype(properties(device_image_scope, host_access_read_write))>
    dev_global;

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class Test>([=]() {
      dev_global[4] = 42; // BUG: out-of-bounds
    });
  }).wait();

  int val;
  Q.copy(dev_global, &val).wait();

  return 0;
}
