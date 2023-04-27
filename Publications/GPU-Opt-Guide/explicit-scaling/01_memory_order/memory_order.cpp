//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef SCALE
#define SCALE  1
#endif

#define N        1024*SCALE
#define SG_SIZE  32

// Number of repetitions
constexpr int repetitions = 16;
constexpr int warm_up_token = -1;

static auto exception_handler = [](sycl::exception_list eList) {
  for (std::exception_ptr const &e : eList) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
      std::cout << "Failure" << std::endl;
      std::terminate();
    }
  }
};

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

#ifdef FLUSH_CACHE
void flush_cache(sycl::queue &q, sycl::buffer<int> &flush_buf) {
  auto flush_size = flush_buf.get_size()/sizeof(int);
  auto ev = q.submit([&](auto &h) {
    sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::noinit);
    h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
  });
  ev.wait_and_throw();
}
#endif

void atomicLatencyTest(sycl::queue &q, sycl::buffer<int> inbuf,
                       sycl::buffer<int> flush_buf, int &res, int iter) {
  const size_t data_size = inbuf.byte_size()/sizeof(int);

  sycl::buffer<int> sum_buf(&res, 1);

  double elapsed = 0;

  for (int k = warm_up_token; k < iter; k++) {
#ifdef FLUSH_CACHE
    flush_cache(q, flush_buf);
#endif

    Timer timer;

    q.submit([&](auto &h) {
      sycl::accessor buf_acc(inbuf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(sycl::nd_range<1>(sycl::range<>{N}, sycl::range<>{SG_SIZE}), [=](sycl::nd_item<1> item)
                                                    [[intel::reqd_sub_group_size(SG_SIZE)]] {
        int i = item.get_global_id(0);
        for (int ii = 0; ii < 1024; ++ii) {
	  auto v =
          #ifdef ATOMIC_RELAXED
             sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>(buf_acc[i]);
          #else
             sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>(buf_acc[i]);
          #endif
          v.fetch_add(1);
        }
      });
    });
    q.wait();
    elapsed += (iter == warm_up_token) ? 0 : timer.Elapsed();
  }
  std::cout << "SUCCESS: Time atomicLatency = " << elapsed << "s" << std::endl;
}

int main(int argc, char *argv[]) {
  sycl::queue q{sycl::gpu_selector_v, exception_handler};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  std::vector<int> data(N);
  std::vector<int> extra(N);

  for (size_t i = 0; i < N ; ++i) {
    data[i]  = 1;
    extra[i] = 1;
  }
  int res=0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  sycl::buffer<int> buf(data.data(), data.size(), props);
  sycl::buffer<int> flush_buf(extra.data(), extra.size(), props);
  atomicLatencyTest(q, buf, flush_buf, res, 16);
}
