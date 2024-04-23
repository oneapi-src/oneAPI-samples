#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/kernel_properties.hpp>

int main() {
  // Creating buffer of 4 ints to be used inside the kernel code
  std::vector<int> input(4);
  sycl::buffer<int> buf(input.data(), 4);
  // Creating SYCL queue
  sycl::queue Queue;

  sycl::range num_items{input.size()};
  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    sycl::accessor buf_acc(buf, cgh, sycl::write_only, sycl::no_init);
    cgh.parallel_for(num_items, [=](auto i) {
      sycl::ext::intel::experimental::set_kernel_properties(
          sycl::ext::intel::experimental::kernel_properties::use_large_grf);
      // Fill buffer with indexes
      buf_acc[i] = i;
    });
  });

  return 0;
}
