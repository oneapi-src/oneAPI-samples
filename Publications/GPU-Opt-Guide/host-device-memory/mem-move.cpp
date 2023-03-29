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

sycl::default_selector d_selector;

template <typename T> using VectorAllocator = AlignedAllocator<T>;

template <typename T> using AlignedVector = std::vector<T, VectorAllocator<T>>;

constexpr size_t array_size = (10 * (1 << 20));

class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

int check_res(AlignedVector<int> &v) {
  for (int i = 0; i < v.size(); i += 2)
    if (v[i] != 24 || v[i + 1] != 2)
      return 0;
  return 1;
}

double myFunc1(sycl::queue &q, AlignedVector<int> &a, AlignedVector<int> &b,
               AlignedVector<int> &c, AlignedVector<int> &d,
               AlignedVector<int> &res, int iter) {
  sycl::range num_items{a.size()};
  VectorAllocator<int> alloc;
  AlignedVector<int> sum(a.size(), alloc);

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer a_buf(a, props);
  sycl::buffer b_buf(b, props);
  sycl::buffer c_buf(b, props);
  sycl::buffer d_buf(b, props);
  sycl::buffer res_buf(res, props);
  sycl::buffer sum_buf(sum.data(), num_items, props);

  Timer timer;
  for (int i = 0; i < iter; i++) {
    // kernel1
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items,
                     [=](auto id) { sum_acc[id] = a_acc[id] + b_acc[id]; });
    });

    {
      sycl::host_accessor h_acc(sum_buf);
      for (int j = 0; j < a.size(); j++)
        if (h_acc[j] > 10)
          h_acc[j] = 1;
        else
          h_acc[j] = 0;
    }

    // kernel2
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor sum_acc(sum_buf, h, sycl::read_only);
      sycl::accessor c_acc(c_buf, h, sycl::read_only);
      sycl::accessor d_acc(d_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor res_acc(res_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items, [=](auto id) {
        res_acc[id] = sum_acc[id] * c_acc[id] + d_acc[id];
      });
    });
    q.wait();
  }
  double elapsed = timer.Elapsed() / iter;
  return (elapsed);
} // end myFunc1

double myFunc2(sycl::queue &q, AlignedVector<int> &a, AlignedVector<int> &b,
               AlignedVector<int> &c, AlignedVector<int> &d,
               AlignedVector<int> &res, int iter) {
  sycl::range num_items{a.size()};
  VectorAllocator<int> alloc;
  AlignedVector<int> sum(a.size(), alloc);

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer a_buf(a, props);
  sycl::buffer b_buf(b, props);
  sycl::buffer c_buf(b, props);
  sycl::buffer d_buf(b, props);
  sycl::buffer res_buf(res, props);
  sycl::buffer sum_buf(sum.data(), num_items, props);

  Timer timer;
  for (int i = 0; i < iter; i++) {
    // kernel1
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items,
                     [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
    });

    // kernel3
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::read_write);
      h.parallel_for(num_items, [=](auto id) {
        if (sum_acc[id] > 10)
          sum_acc[id] = 1;
        else
          sum_acc[id] = 0;
      });
    });

    // kernel2
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor sum_acc(sum_buf, h, sycl::read_only);
      sycl::accessor c_acc(c_buf, h, sycl::read_only);
      sycl::accessor d_acc(d_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor res_acc(res_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items, [=](auto i) {
        res_acc[i] = sum_acc[i] * c_acc[i] + d_acc[i];
      });
    });
    q.wait();
  }
  double elapsed = timer.Elapsed() / iter;
  return (elapsed);
} // end myFunc2

double myFunc3(sycl::queue &q, AlignedVector<int> &a, AlignedVector<int> &b,
               AlignedVector<int> &c, AlignedVector<int> &d,
               AlignedVector<int> &res, int iter) {
  sycl::range num_items{a.size()};
  VectorAllocator<int> alloc;
  AlignedVector<int> sum(a.size(), alloc);

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer a_buf(a, props);
  sycl::buffer b_buf(b, props);
  sycl::buffer c_buf(b, props);
  sycl::buffer d_buf(b, props);
  sycl::buffer res_buf(res, props);
  sycl::buffer sum_buf(sum.data(), num_items, props);

  Timer timer;
  for (int i = 0; i < iter; i++) {
    // kernel1
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items, [=](auto i) {
        int t = a_acc[i] + b_acc[i];
        if (t > 10)
          sum_acc[i] = 1;
        else
          sum_acc[i] = 0;
      });
    });

    // kernel2
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor sum_acc(sum_buf, h, sycl::read_only);
      sycl::accessor c_acc(c_buf, h, sycl::read_only);
      sycl::accessor d_acc(d_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor res_acc(res_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items, [=](auto i) {
        res_acc[i] = sum_acc[i] * c_acc[i] + d_acc[i];
      });
    });
    q.wait();
  }
  double elapsed = timer.Elapsed() / iter;
  return (elapsed);
} // end myFunc3

double myFunc4(sycl::queue &q, AlignedVector<int> &a, AlignedVector<int> &b,
               AlignedVector<int> &c, AlignedVector<int> &d,
               AlignedVector<int> &res, int iter) {
  sycl::range num_items{a.size()};
  VectorAllocator<int> alloc;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer a_buf(a, props);
  sycl::buffer b_buf(b, props);
  sycl::buffer c_buf(b, props);
  sycl::buffer d_buf(b, props);
  sycl::buffer res_buf(res, props);

  Timer timer;
  for (int i = 0; i < iter; i++) {
    // kernel1
    q.submit([&](auto &h) {
      // Input accessors
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor c_acc(c_buf, h, sycl::read_only);
      sycl::accessor d_acc(d_buf, h, sycl::read_only);
      // Output accessor
      sycl::accessor res_acc(res_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(num_items, [=](auto i) {
        int t = a_acc[i] + b_acc[i];
        if (t > 10)
          res_acc[i] = c_acc[i] + d_acc[i];
        else
          res_acc[i] = d_acc[i];
      });
    });
    q.wait();
  }
  double elapsed = timer.Elapsed() / iter;
  return (elapsed);
} // end myFunc4

void InitializeArray(AlignedVector<int> &a) {
  for (size_t i = 0; i < a.size(); i += 2)
    a[i] = 12;
  for (size_t i = 1; i < a.size(); i += 2)
    a[i] = 2;
}

void Initialize(AlignedVector<int> &a) {
  for (size_t i = 0; i < a.size(); i++)
    a[i] = 0;
}

int main() {

  sycl::queue q(d_selector);
  VectorAllocator<int> alloc;
  AlignedVector<int> a(array_size, alloc);
  AlignedVector<int> b(array_size, alloc);
  AlignedVector<int> c(array_size, alloc);
  AlignedVector<int> d(array_size, alloc);
  AlignedVector<int> res(array_size, alloc);

  InitializeArray(a);
  InitializeArray(b);
  InitializeArray(c);
  InitializeArray(d);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";

  // jit the code
  myFunc1(q, a, b, c, d, res, 1);
  // check results
  Initialize(res);
  double elapsed = myFunc1(q, a, b, c, d, res, 1);
  if (check_res(res))
    std::cout << "SUCCESS: Time myFunc1   = " << elapsed << "s\n";
  else
    std::cout << "ERROR: myFunc1 result did not match expected result\n";
  elapsed = myFunc2(q, a, b, c, d, res, 1);
  if (check_res(res))
    std::cout << "SUCCESS: Time myFunc2   = " << elapsed << "s\n";
  else
    std::cout << "ERROR: myFunc1 result did not match expected result\n";
  elapsed = myFunc3(q, a, b, c, d, res, 1);
  if (check_res(res))
    std::cout << "SUCCESS: Time myFunc3   = " << elapsed << "s\n";
  else
    std::cout << "ERROR: myFunc1 result did not match expected result\n";
  elapsed = myFunc4(q, a, b, c, d, res, 1);
  if (check_res(res))
    std::cout << "SUCCESS: Time myFunc4   = " << elapsed << "s\n";
  else
    std::cout << "ERROR: myFunc1 result did not match expected result\n";

  return 0;
}
