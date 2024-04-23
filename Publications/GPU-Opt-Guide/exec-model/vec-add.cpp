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
constexpr size_t array_size = 3 * 5 * 7 * (1 << 17);
typedef std::array<int, array_size> IntArray;

#define mysize (1 << 17)

int VectorAdd1(sycl::queue &q, const IntArray &a, const IntArray &b,
               IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);

  auto start = std::chrono::steady_clock::now();
  auto e = q.submit([&](auto &h) {
    // Input accessors
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    // Output accessor
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(num_items, [=](auto i) {
      for (int j = 0; j < iter; j++)
        sum_acc[i] = a_acc[i] + b_acc[i];
    });
  });
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::cout << "VectorAdd1 completed on device - took " << (end - start).count()
            << " u-secs\n";
  return ((end - start).count());
} // end VectorAdd1

template <int groups>
int VectorAdd2(sycl::queue &q, const IntArray &a, const IntArray &b,
               IntArray &sum, int iter) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  size_t num_groups = groups;
  size_t wg_size = 512;
  // get the max wg_sie instead of 512 size_t wg_size = 512;
  auto start = std::chrono::steady_clock::now();
  q.submit([&](auto &h) {
    // Input accessors
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    // Output accessor
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(
        sycl::nd_range<1>(num_groups * wg_size, wg_size),
        [=](sycl::nd_item<1> index) [[intel::reqd_sub_group_size(32)]] {
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
  std::cout << "VectorAdd2<" << groups << "> completed on device - took "
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

  sycl::queue q(d_selector);

  InitializeArray(a);
  InitializeArray(b);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // check results
  Initialize(sum);
  VectorAdd1(q, a, b, sum, 1);

  for (int i = 0; i < mysize; i++)
    if (sum[i] != 2 * i) {
      std::cout << "add1 Did not match\n";
    }

  Initialize(sum);
  VectorAdd2<1>(q, a, b, sum, 1);
  for (int i = 0; i < mysize; i++)
    if (sum[i] != 2 * i) {
      std::cout << "add2 Did not match\n";
    }

  // time the kernels
  Initialize(sum);
  int t = VectorAdd1(q, a, b, sum, 1000);
  Initialize(sum);
  t = VectorAdd2<1>(q, a, b, sum, 1000);
  t = VectorAdd2<2>(q, a, b, sum, 1000);
  t = VectorAdd2<3>(q, a, b, sum, 1000);
  t = VectorAdd2<4>(q, a, b, sum, 1000);
  t = VectorAdd2<5>(q, a, b, sum, 1000);
  t = VectorAdd2<6>(q, a, b, sum, 1000);
  t = VectorAdd2<7>(q, a, b, sum, 1000);
  t = VectorAdd2<8>(q, a, b, sum, 1000);
  t = VectorAdd2<12>(q, a, b, sum, 1000);
  t = VectorAdd2<16>(q, a, b, sum, 1000);
  t = VectorAdd2<20>(q, a, b, sum, 1000);
  t = VectorAdd2<24>(q, a, b, sum, 1000);
  t = VectorAdd2<28>(q, a, b, sum, 1000);
  t = VectorAdd2<32>(q, a, b, sum, 1000);
  return 0;
}
