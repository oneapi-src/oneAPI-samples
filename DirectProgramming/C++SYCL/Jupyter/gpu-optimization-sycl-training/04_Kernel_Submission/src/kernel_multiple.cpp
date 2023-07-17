//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

// Array type and data size for this example.
constexpr size_t array_size = (1 << 15);
typedef std::array<int, array_size> IntArray;

#define iter 10

int multi_queue(sycl::queue &q, const IntArray &a, const IntArray &b) {
  size_t num_items = a.size();
  IntArray s1, s2, s3;

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf1(s1);
  sycl::buffer sum_buf2(s2);
  sycl::buffer sum_buf3(s3);

  size_t num_groups = 1;
  size_t wg_size = 256;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](sycl::handler &h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf1, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index) {
                       size_t loc_id = index.get_local_id();
                       sum_acc[loc_id] = 0;
                       for (int j = 0; j < 1000; j++)
                         for (size_t i = loc_id; i < array_size; i += wg_size) {
                           sum_acc[loc_id] += a_acc[i] + b_acc[i];
                         }
                     });
    });
    q.submit([&](sycl::handler &h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf2, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index) {
                       size_t loc_id = index.get_local_id();
                       sum_acc[loc_id] = 0;
                       for (int j = 0; j < 1000; j++)
                         for (size_t i = loc_id; i < array_size; i += wg_size) {
                           sum_acc[loc_id] += a_acc[i] + b_acc[i];
                         }
                     });
    });
    q.submit([&](sycl::handler &h) {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf3, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index) {
                       size_t loc_id = index.get_local_id();
                       sum_acc[loc_id] = 0;
                       for (int j = 0; j < 1000; j++)
                         for (size_t i = loc_id; i < array_size; i += wg_size) {
                           sum_acc[loc_id] += a_acc[i] + b_acc[i];
                         }
                     });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "multi_queue completed on device - took "
            << (end - start).count()/ 1e+9 << " seconds\n";
  // check results
  return ((end - start).count());
} // end multi_queue

void InitializeArray(IntArray &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = 1;
}

IntArray a, b;

int main() {

  sycl::queue q;

  InitializeArray(a);
  InitializeArray(b);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // begin in-order submission
  std::cout << "In order queue: Jitting+Execution time\n";
  sycl::queue q1{sycl::property::queue::in_order()};
  multi_queue(q1, a, b);
  std::cout << "In order queue: Execution time\n";
  multi_queue(q1, a, b);
  // end in-order submission

  // begin out-of-order submission
  sycl::queue q2;
  std::cout << "Out of order queue: Jitting+Execution time\n";
  multi_queue(q2, a, b);
  std::cout << "Out of order queue: Execution time\n";
  multi_queue(q2, a, b);
  // end out-of-order submission
  return 0;
}
