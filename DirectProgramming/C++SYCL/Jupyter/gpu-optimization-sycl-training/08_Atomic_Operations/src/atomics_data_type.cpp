//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

constexpr size_t N = 1024 * 100;

int reductionInt(sycl::queue &q, std::vector<int> &data) {
  const size_t data_size = data.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> sum_buf(&sum, 1, props);

  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(data_size, [=](auto index) {
        size_t glob_id = index[0];
        auto v = sycl::atomic_ref<
            int, sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(sum_acc[0]);
        v.fetch_add(buf_acc[glob_id]);
      });
    });
    q.wait();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
  std::cout << "ReductionInt Sum   = " << sum << ", Duration " << (std::chrono::high_resolution_clock::now().time_since_epoch().count() - start) * 1e-9 << " seconds\n";

  return sum;
}

int reductionFloat(sycl::queue &q, std::vector<float> &data) {
  const size_t data_size = data.size();
  float sum = 0.0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<float> buf(data.data(), data_size, props);
  sycl::buffer<float> sum_buf(&sum, 1, props);

  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(data_size, [=](auto index) {
        size_t glob_id = index[0];
        auto v = sycl::atomic_ref<
            float, sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>(sum_acc[0]);
        v.fetch_add(buf_acc[glob_id]);
      });
    });
    q.wait();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
  
  std::cout << "ReductionFloat Sum = " << sum << ", Duration " << (std::chrono::high_resolution_clock::now().time_since_epoch().count() - start) * 1e-9 << " seconds\n";
  return sum;
}

int main(int argc, char *argv[]) {

  sycl::queue q;
  std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";
  {
    std::vector<int> data(N, 1);
    for(int i=0;i<N;i++) data[i] = 1;
    reductionInt(q, data);
  }

  {
    std::vector<float> data(N, 1.0f);
    for(int i=0;i<N;i++) data[i] = 1;
    reductionFloat(q, data);
  }
}
