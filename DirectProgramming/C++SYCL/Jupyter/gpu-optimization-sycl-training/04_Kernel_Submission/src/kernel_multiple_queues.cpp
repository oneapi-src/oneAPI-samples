//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

constexpr int N = 1024;
#define iter 1000

int VectorAdd(sycl::queue &q1, sycl::queue &q2, sycl::queue &q3,
              std::vector<int> a, std::vector<int> b) {

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer<int> *sum_buf[3 * iter];
  for (size_t i = 0; i < (3 * iter); i++)
    sum_buf[i] = new sycl::buffer<int>(256);

  size_t num_groups = 1;
  size_t wg_size = 256;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q1.submit([&](auto &h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      auto sum_acc = sum_buf[3 * i]->get_access<sycl::access::mode::write>(h);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index) {
                       size_t loc_id = index.get_local_id();
                       sum_acc[loc_id] = 0;
                       for (int j = 0; j < 1000; j++)
                         for (size_t i = loc_id; i < N; i += wg_size) {
                           sum_acc[loc_id] += a_acc[i] + b_acc[i];
                         }
                     });
    });
    q2.submit([&](auto &h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      auto sum_acc =
          sum_buf[3 * i + 1]->get_access<sycl::access::mode::write>(h);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index) {
                       size_t loc_id = index.get_local_id();
                       sum_acc[loc_id] = 0;
                       for (int j = 0; j < 1000; j++)
                         for (size_t i = loc_id; i < N; i += wg_size) {
                           sum_acc[loc_id] += a_acc[i] + b_acc[i];
                         }
                     });
    });
    q3.submit([&](auto &h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      auto sum_acc =
          sum_buf[3 * i + 2]->get_access<sycl::access::mode::write>(h);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index) {
                       size_t loc_id = index.get_local_id();
                       sum_acc[loc_id] = 0;
                       for (int j = 0; j < 1000; j++)
                         for (size_t i = loc_id; i < N; i += wg_size) {
                           sum_acc[loc_id] += a_acc[i] + b_acc[i];
                         }
                     });
    });
  }
  q1.wait();
  q2.wait();
  q3.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add completed on device - took " << (end - start).count() / 1e+9 << " seconds\n";
  // check results
  for (size_t i = 0; i < (3 * iter); i++)
    delete sum_buf[i];
  return ((end - start).count());
}


int main() {

  sycl::queue q(sycl::default_selector_v);
  
  std::vector<int> a(N, 1);
  std::vector<int> b(N, 2);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // jit the code
  VectorAdd(q, q, q, a, b);

  std::cout << "\nSubmission to same queue out_of_order\n";
  VectorAdd(q, q, q, a, b);

  sycl::queue q0(sycl::default_selector_v, sycl::property::queue::in_order());
  std::cout << "\nSubmission to same queue in_order\n";
  VectorAdd(q0, q0, q0, a, b);
    
  std::cout << "\nSubmission to different queues with same context\n";
  sycl::queue q1(sycl::default_selector_v);
  sycl::queue q2(q1.get_context(), sycl::default_selector_v);
  sycl::queue q3(q1.get_context(), sycl::default_selector_v);
  VectorAdd(q1, q2, q3, a, b);

  std::cout << "\nSubmission to different queues with different contexts\n";
  sycl::queue q4(sycl::default_selector_v);
  sycl::queue q5(sycl::default_selector_v);
  sycl::queue q6(sycl::default_selector_v);
  VectorAdd(q4, q5, q6, a, b);

  return 0;
}
