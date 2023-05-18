//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include <CL/sycl.hpp>

class SpecializedKernel;

// Identify the specialization constant.
constexpr sycl::specialization_id<int> nx_sc;

int main() {
  sycl::queue queue;

  std::cout << "Running on "
            << queue.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<float> vec(1);
  {
    sycl::buffer<float> buf(vec.data(), vec.size());

    // Application execution stops here asking for input from user
    int Nx;
    std::cout << "Enter input number ..." << std::endl;
    std::cin >> Nx;

    queue.submit([&](sycl::handler &h) {
      sycl::accessor acc(buf, h, sycl::write_only, sycl::no_init);

      // set specialization constant with runtime variable
      h.set_specialization_constant<nx_sc>(Nx);

      h.single_task<SpecializedKernel>([=](sycl::kernel_handler kh) {
        // nx_sc value here will be input value provided at runtime and
        // can be optimized because JIT compiler now treats it as a constant.
        int runtime_const_trip_count = kh.get_specialization_constant<nx_sc>();
        int accum = 0;
        for (int i = 0; i < runtime_const_trip_count; i++) {
          accum = accum + i;
        }
        acc[0] = accum;
      });
    });
  }
  std::cout << vec[0] << std::endl;
  return 0;
}
// Snippet end
