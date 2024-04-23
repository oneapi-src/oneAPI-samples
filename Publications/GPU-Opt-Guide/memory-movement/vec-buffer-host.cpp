//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <array>
#include <chrono>
#include <iostream>

#include <CL/sycl.hpp>

#include "align.hpp"

template <typename T> using VectorAllocator = AlignedAllocator<T>;

template <typename T> using AlignedVector = std::vector<T, VectorAllocator<T>>;

constexpr size_t array_size = (1 << 15);

// Snippet1 Begin
int VectorAdd0(sycl::queue &q, AlignedVector<int> &a, AlignedVector<int> &b,
               AlignedVector<int> &sum, int iter) {
  sycl::range num_items{a.size()};

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  for (int i = 0; i < iter; i++) {
    sycl::buffer a_buf(a, props);
    sycl::buffer b_buf(b, props);
    sycl::buffer sum_buf(sum.data(), num_items, props);
    {
      sycl::host_accessor a_host_acc(a_buf);
      std::cout << "add0: buff memory address =" << a_host_acc.get_pointer()
                << "\n";
      std::cout << "add0: address of vector a = " << a.data() << "\n";
    }
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      sycl::stream out(1024 * 1024, 1 * 128, h);

      h.parallel_for(num_items, [=](auto i) {
        if (i[0] == 0)
          out << "add0:  dev addr = " << a_acc.get_pointer() << "\n";
        sum_acc[i] = a_acc[i] + b_acc[i];
      });
    });
  }
  q.wait();
  return (0);
}
// Snippet1 End

// Snippet2 Begin
int VectorAdd1(sycl::queue &q, const AlignedVector<int> &a,
               const AlignedVector<int> &b, AlignedVector<int> &sum, int iter) {
  sycl::range num_items{a.size()};

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  for (int i = 0; i < iter; i++) {
    sycl::buffer a_buf(a, props);
    sycl::buffer b_buf(b, props);
    sycl::buffer sum_buf(sum.data(), num_items, props);
    {
      sycl::host_accessor a_host_acc(a_buf);
      std::cout << "add1: buff memory address =" << a_host_acc.get_pointer()
                << "\n";
      std::cout << "add1: address of vector aa = " << a.data() << "\n";
    }
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      sycl::stream out(16 * 1024, 16 * 1024, h);

      h.parallel_for(num_items, [=](auto i) {
        if (i[0] == 0)
          out << "add1: dev addr = " << a_acc.get_pointer() << "\n";
        sum_acc[i] = a_acc[i] + b_acc[i];
      });
    });
  }
  q.wait();
  return (0);
}
// Snippet2 End

// Snippet3 Begin
int VectorAdd2(sycl::queue &q, AlignedVector<int> &a, AlignedVector<int> &b,
               AlignedVector<int> &sum, int iter) {
  sycl::range num_items{a.size()};

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    sycl::buffer a_buf(a, props);
    sycl::buffer b_buf(b, props);
    sycl::buffer sum_buf(sum.data(), num_items, props);
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items,
                     [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add2 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
}
// Snippet3 End

// Snippet4 Begin
int VectorAdd3(sycl::queue &q, const AlignedVector<int> &a,
               const AlignedVector<int> &b, AlignedVector<int> &sum, int iter) {
  sycl::range num_items{a.size()};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
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
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add3 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
}
// Snippet4 End

void InitializeArray(AlignedVector<int> &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = i;
}

void Initialize(AlignedVector<int> &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = 0;
}

int main() {

  sycl::queue q(sycl::default_selector_v);
  VectorAllocator<int> alloc;
  AlignedVector<int> a(array_size, alloc);
  AlignedVector<int> b(array_size, alloc);
  AlignedVector<int> sum(array_size, alloc);

  InitializeArray(a);
  InitializeArray(b);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // jit the code
  VectorAdd1(q, a, b, sum, 1);
  // check results
  Initialize(sum);
  VectorAdd1(q, a, b, sum, 1);

  for (size_t i = 0; i < a.size(); i++)
    if (sum[i] != static_cast<int>(2 * i)) {
      std::cout << "add1 Did not match\n";
    }

  Initialize(sum);
  VectorAdd0(q, a, b, sum, 1);
  for (size_t i = 0; i < a.size(); i++)
    if (sum[i] != static_cast<int>(2 * i)) {
      std::cout << "add0 Did not match\n";
    }

  Initialize(sum);
  VectorAdd2(q, a, b, sum, 1000);
  Initialize(sum);
  VectorAdd3(q, a, b, sum, 1000);
  return 0;
}
