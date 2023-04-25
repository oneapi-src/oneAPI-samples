//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include "align.hpp"
#include <CL/sycl.hpp>
#include <array>
#include <chrono>
#include <iostream>

template <typename T> using VectorAllocator = AlignedAllocator<T>;

template <typename T> using AlignedVector = std::vector<T, VectorAllocator<T>>;

constexpr size_t array_size = (1 << 15);

template <typename T> void InitializeArray(AlignedVector<T> &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = (T)i * (T)i;
}

template <typename T> void Initialize(AlignedVector<T> &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = 0;
}

// Snippet1 Begin
//
int VectorInt(sycl::queue &q, int iter) {
  VectorAllocator<int> alloc;
  AlignedVector<int> a(array_size, alloc);
  AlignedVector<int> b(array_size, alloc);

  InitializeArray<int>(a);
  InitializeArray<int>(b);
  sycl::range num_items{a.size()};
  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](sycl::handler &h) {
      // InpuGt accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_write);
      sycl::accessor b_acc(a_buf, h, sycl::read_only);

      h.parallel_for(num_items, [=](auto i) {
        auto v = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            a_acc[0]);
        v += b_acc[i];
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector int completed on device - took " << (end - start).count()
            << " u-secs\n";
  return ((end - start).count());
}
// Snippet1 End
//

// Snippet2 Begin
//
int VectorFloat(sycl::queue &q, int iter) {
  VectorAllocator<float> alloc;
  AlignedVector<float> a(array_size, alloc);
  AlignedVector<float> b(array_size, alloc);

  InitializeArray<float>(a);
  InitializeArray<float>(b);
  sycl::range num_items{a.size()};
  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](sycl::handler &h) {
      // InpuGt accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_write);
      sycl::accessor b_acc(a_buf, h, sycl::read_only);

      h.parallel_for(num_items, [=](auto i) {
        auto v = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            a_acc[0]);
        v += b_acc[i];
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector float completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
}
// Snippet2 End
//
// Snippet3 Begin
//
int VectorDouble(sycl::queue &q, int iter) {
  VectorAllocator<double> alloc;
  AlignedVector<double> a(array_size, alloc);
  AlignedVector<double> b(array_size, alloc);

  InitializeArray<double>(a);
  InitializeArray<double>(b);
  sycl::range num_items{a.size()};
  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](sycl::handler &h) {
      // InpuGt accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_write);
      sycl::accessor b_acc(a_buf, h, sycl::read_only);

      h.parallel_for(num_items, [=](auto i) {
        auto v = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            a_acc[0]);
        v += b_acc[i];
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector Double completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
}
// Snippet3 End

int main() {

  sycl::queue q(sycl::gpu_selector_v);
  VectorAllocator<int> alloc;
  AlignedVector<int> a(array_size, alloc);
  AlignedVector<int> b(array_size, alloc);

  InitializeArray<int>(a);
  InitializeArray<int>(b);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  VectorInt(q, 10);

  VectorInt(q, 10);
  VectorFloat(q, 10);
  VectorDouble(q, 10);

  return 0;
}
