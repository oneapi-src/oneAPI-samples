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

#define mysize (1 << 17)

size_t VectorAdd(sycl::queue &q, const IntArray &a, const IntArray &b,
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
  std::cout << "Vector add completed on device - took " << (end - start).count()
            << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd

size_t VectorAdd1(sycl::queue &q, const IntArray &a, const IntArray &b,
                  IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  size_t num_groups = 1;
  size_t wg_size = 16;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index)
                         [[intel::reqd_sub_group_size(16)]] {
                           // no unrolling
                           size_t loc_id = index.get_local_id();
                           for (size_t i = loc_id; i < mysize; i += wg_size) {
                             sum_acc[i] = a_acc[i] + b_acc[i];
                           }
                         });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add1 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd1

size_t VectorAdd2(sycl::queue &q, const IntArray &a, const IntArray &b,
                  IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  // size_t num_groups =
  // q.get_device().get_info<sycl::info::device::max_compute_units>(); size_t
  // wg_size =
  // q.get_device().get_info<sycl::info::device::max_work_group_size>();
  size_t num_groups = 1;
  size_t wg_size = 16;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index) [[intel::reqd_sub_group_size(
                         16)]] [[intel::kernel_args_restrict]] {
                       size_t loc_id = index.get_local_id();
        // unroll with a directive
#pragma unroll(2)
                       for (size_t i = loc_id; i < mysize; i += wg_size) {
                         sum_acc[i] = a_acc[i] + b_acc[i];
                       }
                     });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add2 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd2

size_t VectorAdd3(sycl::queue &q, const IntArray &a, const IntArray &b,
                  IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  size_t num_groups = 1;
  size_t wg_size = 16;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index)
                         [[intel::reqd_sub_group_size(16)]] {
                           // Manual unrolling
                           size_t loc_id = index.get_local_id();
                           for (size_t i = loc_id; i < mysize; i += 32) {
                             sum_acc[i] = a_acc[i] + b_acc[i];
                             sum_acc[i + 16] = a_acc[i + 16] + b_acc[i + 16];
                           }
                         });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add3 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd3

size_t VectorAdd4(sycl::queue &q, const IntArray &a, const IntArray &b,
                  IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  size_t num_groups = 1;
  size_t wg_size = 16;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index)
                         [[intel::reqd_sub_group_size(16)]] {
                           // Manual unrolling
                           size_t loc_id = index.get_local_id();
                           for (size_t i = loc_id; i < mysize; i += 32) {
                             int t1 = a_acc[i];
                             int t2 = b_acc[i];
                             int t3 = a_acc[i + 16];
                             int t4 = b_acc[i + 16];
                             sum_acc[i] = t1 + t2;
                             sum_acc[i + 16] = t3 + t4;
                           }
                         });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add4 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd4

size_t VectorAdd5(sycl::queue &q, const IntArray &a, const IntArray &b,
                  IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  size_t num_groups = 1;
  size_t wg_size = 16;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                     [=](sycl::nd_item<1> index) [[intel::reqd_sub_group_size(
                         16)]] [[intel::kernel_args_restrict]] {
                       // compiler needs to hoist the loads
                       size_t loc_id = index.get_local_id();
                       for (size_t i = loc_id; i < mysize; i += 32) {
                         sum_acc[i] = a_acc[i] + b_acc[i];
                         sum_acc[i + 16] = a_acc[i + 16] + b_acc[i + 16];
                       }
                     });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "Vector add5 completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd5

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

  InitializeArray(a);
  InitializeArray(b);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // jit the code
  VectorAdd(q, a, b, sum, 1000);
  // check results
  Initialize(sum);
  VectorAdd1(q, a, b, sum, 1);

  for (int i = 0; i < mysize; i++)
    if (sum[i] != 2 * i) {
      std::cout << "add1 Did not match\n";
    }

  Initialize(sum);
  VectorAdd2(q, a, b, sum, 1);
  for (int i = 0; i < mysize; i++)
    if (sum[i] != 2 * i) {
      std::cout << "add2 Did not match\n";
    }
  // time the kernels
  Initialize(sum);
  VectorAdd3(q, a, b, sum, 1);
  for (int i = 0; i < mysize; i++)
    if (sum[i] != 2 * i) {
      std::cout << "add3 Did not match "
                << "sum[" << i << "]=" << sum[i] << "\n";
    }
  Initialize(sum);
  VectorAdd5(q, a, b, sum, 1);
  for (int i = 0; i < mysize; i++)
    if (sum[i] != 2 * i) {
      std::cout << "add5 Did not match "
                << "sum[" << i << "]=" << sum[i] << "\n";
    }
  Initialize(sum);
  VectorAdd1(q, a, b, sum, 1000);
  Initialize(sum);
  VectorAdd2(q, a, b, sum, 1000);
  Initialize(sum);
  VectorAdd3(q, a, b, sum, 1000);
  Initialize(sum);
  VectorAdd4(q, a, b, sum, 1000);
  Initialize(sum);
  VectorAdd5(q, a, b, sum, 1000);
  return 0;
}
