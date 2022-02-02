// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;
using dinfo = info::device;
constexpr int N = 42;

template <typename T> void foo(T data, id<1> i) { data[i] = N; }

int main() {
  queue Q;
  auto dev = Q.get_device();
  auto ctxt = Q.get_context();
  bool usm_shared = dev.get_info<dinfo::usm_shared_allocations>();
  bool usm_device = dev.get_info<dinfo::usm_device_allocations>();
  bool use_USM = usm_shared || usm_device;

  if (use_USM) {
    int *data;
    if (usm_shared) {
      data = malloc_shared<int>(N, Q);
    } else /* use device allocations */ {
      data = malloc_device<int>(N, Q);
    }
    std::cout << "Using USM with "
              << ((get_pointer_type(data, ctxt) == usm::alloc::shared)
                  ? "shared"
                  : "device")
              << " allocations on "
              << get_pointer_device(data, ctxt).get_info<dinfo::name>()
              << "\n";
    Q.parallel_for(N, [=](id<1> i) { foo(data, i); });
    Q.wait();
    free(data, Q);
  } else /* use buffers */ {
    buffer<int, 1> data{range{N}};
    Q.submit([&](handler &h) {
        accessor a(data, h);
        h.parallel_for(N, [=](id<1> i) { foo(a, i); });
      });
    Q.wait();
  }
  return 0;
}
