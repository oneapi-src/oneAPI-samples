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
constexpr size_t array_size = 3 * 5 * 7 * (1 << 17);
typedef std::array<int, array_size> IntArray;

// #define mysize (1 << 17)

// Executing entire kernel on the GPU
size_t VectorAdd1(sycl::queue &q, const IntArray &a, const IntArray &b,
                  IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {

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
  std::cout << "Vector add1 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd1

// Executing half on GPU and the other half on CPU
size_t VectorAdd2(sycl::queue &q1, sycl::queue &q2, const IntArray &a,
                  const IntArray &b, IntArray &sum, int iter) {
  sycl::range num_items{a.size() / 2};

  auto start = std::chrono::steady_clock::now();
  {
    sycl::buffer a1_buf(a.data(), num_items);
    sycl::buffer b1_buf(b.data(), num_items);
    sycl::buffer sum1_buf(sum.data(), num_items);

    sycl::buffer a2_buf(a.data() + a.size() / 2, num_items);
    sycl::buffer b2_buf(b.data() + a.size() / 2, num_items);
    sycl::buffer sum2_buf(sum.data() + a.size() / 2, num_items);
    for (int i = 0; i < iter; i++) {

      q1.submit([&](auto &h) {
        // Input accessors
        sycl::accessor a_acc(a1_buf, h, sycl::read_only);
        sycl::accessor b_acc(b1_buf, h, sycl::read_only);
        // Output accessor
        sycl::accessor sum_acc(sum1_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(num_items,
                       [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
      });
      // do the work on host
      q2.submit([&](auto &h) {
        // Input accessors
        sycl::accessor a_acc(a2_buf, h, sycl::read_only);
        sycl::accessor b_acc(b2_buf, h, sycl::read_only);
        // Output accessor
        sycl::accessor sum_acc(sum2_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(num_items,
                       [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
      });
    }
    // On some platforms this explicit flush of queues is needed
    // to ensure the overlap in execution between the CPU and GPU
    // cl_command_queue cq = q1.get();
    // clFlush(cq);
    // cq=q2.get();
    // clFlush(cq);
  }
  q1.wait();
  q2.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add2 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd2

void InitializeArray(IntArray &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = i;
}

void Initialize(IntArray &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = 0;
}
IntArray a, b, sum;

int main() {

  sycl::queue q(sycl::default_selector_v);
  sycl::queue q1(sycl::gpu_selector_v);
  sycl::queue q2(sycl::cpu_selector_v);

  InitializeArray(a);
  InitializeArray(b);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // jit the code
  VectorAdd1(q, a, b, sum, 10);
  // check results
  Initialize(sum);
  VectorAdd1(q, a, b, sum, 1);

  for (size_t i = 0; i < array_size; i++)
    if (sum[i] != static_cast<int>(2 * i)) {
      std::cout << "add1 Did not match\n";
    }

  Initialize(sum);
  VectorAdd2(q1, q2, a, b, sum, 1);
  for (size_t i = 0; i < array_size; i++)
    if (sum[i] != static_cast<int>(2 * i)) {
      std::cout << "add2 Did not match\n";
    }
  Initialize(sum);
  VectorAdd1(q, a, b, sum, 10);
  Initialize(sum);
  VectorAdd2(q1, q2, a, b, sum, 10);
  return 0;
}
