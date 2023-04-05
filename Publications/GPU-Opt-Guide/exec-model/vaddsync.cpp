//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <array>
#include <chrono>
#include <iostream>

auto d_selector = sycl::default_selector_v;

// Array type and data size for this example.
constexpr size_t array_size = 3 * 5 * 7 * (1 << 18);
typedef std::array<int, array_size> IntArray;

#define mysize (1 << 18)

// VectorAdd3
template <int groups, int wg_size, int sg_size>
int VectorAdd3(sycl::queue &q, const IntArray &a, const IntArray &b,
               IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  size_t num_groups = groups;
  auto start = std::chrono::steady_clock::now();
  q.submit([&](auto &h) {
    // Input accessors
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    // Output accessor
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(
        sycl::nd_range<1>(num_groups * wg_size, wg_size),
        [=](sycl::nd_item<1> index) [[intel::reqd_sub_group_size(sg_size)]] {
          size_t grp_id = index.get_group()[0];
          size_t loc_id = index.get_local_id();
          size_t start = grp_id * mysize;
          size_t end = start + mysize;
          for (int j = 0; j < iter; j++)
            for (size_t i = start + loc_id; i < end; i += wg_size) {
              sum_acc[i] = a_acc[i] + b_acc[i];
            }
        });
  });
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "VectorAdd3<" << groups << "> completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd3

// VectorAdd4
template <int groups, int wg_size, int sg_size>
int VectorAdd4(sycl::queue &q, const IntArray &a, const IntArray &b,
               IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  size_t num_groups = groups;
  auto start = std::chrono::steady_clock::now();
  q.submit([&](auto &h) {
    // Input accessors
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    // Output accessor
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(
        sycl::nd_range<1>(num_groups * wg_size, wg_size),
        [=](sycl::nd_item<1> index) [[intel::reqd_sub_group_size(sg_size)]] {
          index.barrier(sycl::access::fence_space::local_space);
          size_t grp_id = index.get_group()[0];
          size_t loc_id = index.get_local_id();
          size_t start = grp_id * mysize;
          size_t end = start + mysize;
          for (int j = 0; j < iter; j++) {
            for (size_t i = start + loc_id; i < end; i += wg_size) {
              sum_acc[i] = a_acc[i] + b_acc[i];
            }
          }
        });
  });
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "VectorAdd4<" << groups << "> completed on device - took "
            << (end - start).count() << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd4

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

  sycl::queue q(d_selector);

  InitializeArray(a);
  InitializeArray(b);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // check results
  Initialize(sum);
  VectorAdd3<6, 320, 8>(q, a, b, sum, 1);

  for (int i = 0; i < mysize; i++)
    if (sum[i] != 2 * i) {
      std::cout << "add3 Did not match\n";
    }

  Initialize(sum);
  VectorAdd4<6, 320, 8>(q, a, b, sum, 1);
  for (int i = 0; i < mysize; i++)
    if (sum[i] != 2 * i) {
      std::cout << "add4 Did not match\n";
    }

  // group1
  Initialize(sum);
  VectorAdd3<8, 320, 8>(q, a, b, sum, 10000);
  Initialize(sum);
  VectorAdd4<8, 320, 8>(q, a, b, sum, 10000);
  // end group1

  // group2
  Initialize(sum);
  VectorAdd3<24, 224, 8>(q, a, b, sum, 10000);
  Initialize(sum);
  VectorAdd4<24, 224, 8>(q, a, b, sum, 10000);
  // end group2
  return 0;
}
