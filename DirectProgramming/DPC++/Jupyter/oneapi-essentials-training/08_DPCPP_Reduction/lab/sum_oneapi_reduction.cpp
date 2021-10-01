#include <CL/sycl.hpp>

using namespace sycl;

static const size_t N = 1024; // global size
static const size_t B = 256; // work-group size

int main() {
  //# setup queue with in_order property
  queue q(property::queue::in_order{});
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;

  //# initialize tmp array with values from 1 to N
  int tmp[N];
  for (int i = 0; i < N; i++) tmp[i] = i;
  //# initialize data array using usm
  int *data = malloc_device<int>(N, q);

  //# move tmp from host to device
  q.memcpy(data, tmp, sizeof(int)*N);

  //# implicit USM for writing sum value
  int* sum = malloc_shared<int>(1, q);
  *sum = 0;

  //# nd-range kernel parallel_for with reduction parameter
  q.submit([&](handler& h) {
     h.parallel_for(nd_range<1>{N, B}, ONEAPI::reduction(sum, ONEAPI::plus<>()), [=](nd_item<1> it, auto& sum) {
       int i = it.get_global_id(0);
       sum += data[i];
     });
   }).wait();

  std::cout << "Sum = " << *sum << std::endl;

  free(data, q);
  free(sum, q);
  return 0;
}
