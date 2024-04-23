//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include <CL/sycl.hpp>
#include <vector>

class specialized_kernel;

// const static identifier of specialization constant
const static sycl::specialization_id<float> value_id;

// Fetch a value at runtime.
float get_value() { return 10; };

int main() {
  sycl::queue queue;

  std::vector<float> vec(1);
  {
    sycl::buffer<float> buffer(vec.data(), vec.size());
    queue.submit([&](auto &cgh) {
      sycl::accessor acc(buffer, cgh, sycl::write_only, sycl::no_init);

      // Set value of specialization constant.
      cgh.template set_specialization_constant<value_id>(get_value());

      // Runtime builds the kernel with specialization constant
      // replaced by the literal value provided in the preceding
      // call of `set_specialization_constant<value_id>`
      cgh.template single_task<specialized_kernel>(
          [=](sycl::kernel_handler kh) {
            const float val = kh.get_specialization_constant<value_id>();
            acc[0] = val;
          });
    });
  }
  queue.wait_and_throw();

  std::cout << vec[0] << std::endl;

  return 0;
}
// Snippet end
