//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <array>
#include <chrono>
#include <iostream>

// Array type and data size for this example.
constexpr size_t array_size = (1 << 15);
typedef std::array<int, array_size> IntArray;

#define iter 10

int VectorAdd(sycl::queue &q1, sycl::queue &q2, sycl::queue &q3,
              const IntArray &a, const IntArray &b) {

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
                       for (size_t i = loc_id; i < array_size; i += wg_size) {
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
                       for (size_t i = loc_id; i < array_size; i += wg_size) {
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
                       for (size_t i = loc_id; i < array_size; i += wg_size) {
                         sum_acc[loc_id] += a_acc[i] + b_acc[i];
                       }
                     });
    });
  }
  q1.wait();
  q2.wait();
  q3.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add completed on device - took " << (end - start).count()
            << " u-secs\n";
  // check results
  for (size_t i = 0; i < (3 * iter); i++)
    delete sum_buf[i];
  return ((end - start).count());
} // end VectorAdd

void InitializeArray(IntArray &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = 1;
}

void Initialize(IntArray &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = 0;
}
IntArray a, b;

int main() {

  sycl::queue q(sycl::default_selector_v);

  InitializeArray(a);
  InitializeArray(b);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // jit the code
  VectorAdd(q, q, q, a, b);

  std::cout << "Submission to same queue\n";
  // Submission to same queue
  VectorAdd(q, q, q, a, b);
  // End Submission to same queue

  std::cout << "Submission to different queues with same context\n";
  // Submission to different queues with same context
  sycl::queue q1(sycl::default_selector_v);
  sycl::queue q2(q1.get_context(), sycl::default_selector_v);
  sycl::queue q3(q1.get_context(), sycl::default_selector_v);
  VectorAdd(q1, q2, q3, a, b);
  // End Submission to different queues with same context

  std::cout << "Submission to diffferent queues with different contexts\n";
  // Submission to different queues with different contexts
  sycl::queue q4(sycl::default_selector_v);
  sycl::queue q5(sycl::default_selector_v);
  sycl::queue q6(sycl::default_selector_v);
  VectorAdd(q4, q5, q6, a, b);
  // End Submission to different queues with different contexts

  return 0;
}
