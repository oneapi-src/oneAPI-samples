//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

#define N 13762560
const int WG_SIZE = 256; // 256,512,1024
const int SG_SIZE = 32;  // 8, 16, 32

template <int groups, int wg_size, int sg_size>
int VectorAdd(sycl::queue &q, std::vector<int> &a, std::vector<int> &b,
               std::vector<int> &sum) {
  sycl::range num_items{a.size()};

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf(sum.data(), num_items);
  size_t num_groups = groups;

  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  q.submit([&](auto &h) {
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(
        sycl::nd_range<1>(num_groups * wg_size, wg_size), [=
    ](sycl::nd_item<1> index) [[intel::reqd_sub_group_size(sg_size)]] {
          size_t grp_id = index.get_group()[0];
          size_t loc_id = index.get_local_id();
          size_t start = grp_id * N;
          size_t end = start + N;
          for (size_t i = start + loc_id; i < end; i += wg_size) {
            sum_acc[i] = a_acc[i] + b_acc[i];
          }
        });
  });
  q.wait();
  auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::cout << "VectorAdd<" << groups << "> completed on device - "
            << (end - start) * 1e-9 << " seconds\n";
  return 0;
}

int main() {

  sycl::queue q;

  std::vector<int> a(N), b(N), sum(N);
  for (size_t i = 0; i < a.size(); i++){
    a[i] = i;
    b[i] = i;
    sum[i] = 0;
  }

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "Vector size: " << a.size() << "\n";
    
  VectorAdd<1,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<2,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<3,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<4,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<5,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<6,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<7,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<8,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<12,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<16,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<20,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<24,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<28,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<32,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<40,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<48,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<56,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<64,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<80,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<96,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<112,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<128,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<192,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<256,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<384,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<512,WG_SIZE,SG_SIZE>(q, a, b, sum);
  VectorAdd<1024,WG_SIZE,SG_SIZE>(q, a, b, sum);
    
  return 0;
}
