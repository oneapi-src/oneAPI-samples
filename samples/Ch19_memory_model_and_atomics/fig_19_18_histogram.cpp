// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>

using namespace sycl;

template <typename T, int dimensions>
using local_accessor =
    accessor<T, dimensions, access::mode::read_write, access::target::local>;

std::tuple<size_t, size_t> distribute_range(group<1> g, size_t N) {
  size_t work_per_group = N / g.get_group_range(0);
  size_t remainder = N - g.get_group_range(0) * work_per_group;
  size_t group_start =
      g.get_id(0) * work_per_group + min(g.get_id(0), remainder);
  size_t group_end =
      (g.get_id(0) + 1) * work_per_group + min(g.get_id(0) + 1, remainder);
  return {group_start, group_end};
}

// Define shorthand aliases for the types of atomic needed by this kernel
namespace {
  using memory_order = memory_order;
  using memory_scope = memory_scope;

  template <typename T>
  using local_atomic_ref = atomic_ref<
    T,
    memory_order::relaxed,
    memory_scope::work_group,
    access::address_space::local_space>;

  template <typename T>
  using global_atomic_ref = atomic_ref<
    T,
    memory_order::relaxed,
    memory_scope::system,
    access::address_space::global_space>;
}

int main() {
  queue Q;

  uint32_t num_groups = 72;
  uint32_t num_items = 16;

  size_t N = 1024;
  size_t B = 64;
  uint32_t* input = malloc_shared<uint32_t>(N, Q);
  uint32_t* histogram = malloc_shared<uint32_t>(B, Q);
  std::generate(input, input + N, std::mt19937{});
  std::fill(histogram, histogram + B, 0);

  Q.submit([&](handler& h) {
     auto local = local_accessor<uint32_t, 1>{B, h};
     h.parallel_for(
         nd_range<1>{num_groups * num_items, num_items}, [=](nd_item<1> it) {
           // Phase 1: Work-items co-operate to zero local memory
           for (int32_t b = it.get_local_id(0); b < B;
                b += it.get_local_range(0)) {
             local[b] = 0;
           }
           it.barrier(); // Wait for all to be zeroed

           // Phase 2: Work-groups each compute a chunk of the input
           // Work-items co-operate to compute histogram in local memory
           auto grp = it.get_group();
           const auto [group_start, group_end] = distribute_range(grp, N);
           for (int i = group_start + it.get_local_id(0); i < group_end;
                i += it.get_local_range(0)) {
             int32_t b = input[i] % B;
             local_atomic_ref<uint32_t>(local[b])++;
           }
           it.barrier(); // Wait for all local histogram updates to complete

           // Phase 3: Work-items co-operate to update global memory
           for (int32_t b = it.get_local_id(0); b < B;
                b += it.get_local_range(0)) {
             global_atomic_ref<uint32_t>(histogram[b]) += local[b];
           }
         });
   }).wait();

  // Compute reference histogram serially on the host
  bool passed = true;
  uint32_t* gold = static_cast<uint32_t*>(malloc(B * sizeof(uint32_t)));
  std::fill(gold, gold + B, 0);
  for (int i = 0; i < N; ++i) {
    uint32_t b = input[i] % B;
    gold[b]++;
  }
  for (int b = 0; b < B; ++b) {
    if (gold[b] != histogram[b]) {
      passed = false;
    }
  }
  std::cout << ((passed) ? "SUCCESS\n" : "FAILURE\n");

  free(gold);
  free(histogram, Q);
  free(input, Q);
  return (passed) ? 0 : 1;
}
