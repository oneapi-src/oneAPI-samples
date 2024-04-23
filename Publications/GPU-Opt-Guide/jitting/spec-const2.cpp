//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include <CL/sycl.hpp>
#include <chrono>
#include <vector>

class specialized_kernel;
class literal_kernel;

// const static identifier of specialization constant
const static sycl::specialization_id<float> value_id;

// Fetch a value at runtime.
float get_value() { return 10; };

int main() {
  sycl::queue queue;

  // Get kernel ID from kernel class qualifier
  sycl::kernel_id specialized_kernel_id =
      sycl::get_kernel_id<specialized_kernel>();

  // Construct kernel bundle with only specialized_kernel in the input state
  sycl::kernel_bundle kb_src =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          queue.get_context(), {specialized_kernel_id});
  // set specialization constant value
  kb_src.set_specialization_constant<value_id>(get_value());

  auto start = std::chrono::steady_clock::now();
  // build the kernel bundle for the set value
  sycl::kernel_bundle kb_exe = sycl::build(kb_src);
  auto end = std::chrono::steady_clock::now();
  std::cout << "specialization took - " << (end - start).count()
            << " nano-secs\n";

  std::vector<float> vec{0, 0, 0, 0, 0};
  sycl::buffer<float> buffer1(vec.data(), vec.size());
  sycl::buffer<float> buffer2(vec.data(), vec.size());
  start = std::chrono::steady_clock::now();
  {
    queue.submit([&](auto &cgh) {
      sycl::accessor acc(buffer1, cgh, sycl::write_only, sycl::no_init);

      // use the precompiled kernel bundle in the executable state
      cgh.use_kernel_bundle(kb_exe);

      cgh.template single_task<specialized_kernel>(
          [=](sycl::kernel_handler kh) {
            float v = kh.get_specialization_constant<value_id>();
            acc[0] = v;
          });
    });
    queue.wait_and_throw();
  }
  end = std::chrono::steady_clock::now();

  {
    sycl::host_accessor host_acc(buffer1, sycl::read_only);
    std::cout << "result1 (c): " << host_acc[0] << " " << host_acc[1] << " "
              << host_acc[2] << " " << host_acc[3] << " " << host_acc[4]
              << std::endl;
  }
  std::cout << "execution took : " << (end - start).count() << " nano-secs\n";

  start = std::chrono::steady_clock::now();
  {
    queue.submit([&](auto &cgh) {
      sycl::accessor acc(buffer2, cgh, sycl::write_only, sycl::no_init);
      cgh.template single_task<literal_kernel>([=]() { acc[0] = 20; });
    });
    queue.wait_and_throw();
  }
  end = std::chrono::steady_clock::now();

  {
    sycl::host_accessor host_acc(buffer2, sycl::read_only);
    std::cout << "result2 (c): " << host_acc[0] << " " << host_acc[1] << " "
              << host_acc[2] << " " << host_acc[3] << " " << host_acc[4]
              << std::endl;
  }
  std::cout << "execution took - " << (end - start).count() << " nano-secs\n";
}
// Snippet end
