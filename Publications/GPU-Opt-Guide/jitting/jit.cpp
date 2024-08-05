//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// Snippet begin
#include <CL/sycl.hpp>
#include <array>
#include <chrono>
#include <iostream>

// Array type and data size for this example.
constexpr size_t array_size = (1 << 16);
typedef std::array<int, array_size> IntArray;

void VectorAdd1(sycl::queue &q, const IntArray &a, const IntArray &b,
                IntArray &sum) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);

  auto e = q.submit([&](auto &h) {
    // Input accessors
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    // Output accessor
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(num_items,
                   [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
  });
  q.wait();
}

void VectorAdd2(sycl::queue &q, const IntArray &a, const IntArray &b,
                IntArray &sum) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);

  auto e = q.submit([&](auto &h) {
    // Input accessors
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    // Output accessor
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(num_items,
                   [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
  });
  q.wait();
}

void InitializeArray(IntArray &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = i;
}

int main() {
  IntArray a, b, sum;

  InitializeArray(a);
  InitializeArray(b);

  sycl::queue q(sycl::default_selector_v,
                sycl::property::queue::enable_profiling{});

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";
  auto start = std::chrono::steady_clock::now();
  VectorAdd1(q, a, b, sum);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Initial Vector add1 successfully completed on device - took "
            << (end - start).count() << " nano-secs\n";

  start = std::chrono::steady_clock::now();
  VectorAdd1(q, a, b, sum);
  end = std::chrono::steady_clock::now();
  std::cout << "Second Vector add1 successfully completed on device - took "
            << (end - start).count() << " nano-secs\n";

  start = std::chrono::steady_clock::now();
  VectorAdd2(q, a, b, sum);
  end = std::chrono::steady_clock::now();
  std::cout << "Initial Vector add2 successfully completed on device - took "
            << (end - start).count() << " nano-secs\n";

  start = std::chrono::steady_clock::now();
  VectorAdd2(q, a, b, sum);
  end = std::chrono::steady_clock::now();
  std::cout << "Second Vector add2 successfully completed on device - took "
            << (end - start).count() << " nano-secs\n";
  return 0;
}
// Snippet end
