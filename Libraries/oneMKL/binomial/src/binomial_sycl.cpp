//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <cstdio>
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include "binomial.hpp"

constexpr int sg_size = 32;
constexpr int wg_size = 128;

sycl::queue* binomial_queue;

Binomial::Binomial() {
  binomial_queue = new sycl::queue;

  h_call_result = sycl::malloc_shared<DATA_TYPE>(opt_n, *binomial_queue);
  h_stock_price = sycl::malloc_shared<DATA_TYPE>(opt_n, *binomial_queue);
  h_option_strike = sycl::malloc_shared<DATA_TYPE>(opt_n, *binomial_queue);
  h_option_years = sycl::malloc_shared<DATA_TYPE>(opt_n, *binomial_queue);

  binomial_queue->fill(h_call_result, DATA_TYPE(0), opt_n);

  constexpr int rand_seed = 777;
  namespace mkl_rng = oneapi::mkl::rng;

  // create random number generator object
  mkl_rng::philox4x32x10 engine(
#if !INIT_ON_HOST
      *binomial_queue,
#else
      sycl::queue{sycl::cpu_selector_v},
#endif
      rand_seed);

  sycl::event event_1 = mkl_rng::generate(
      mkl_rng::uniform<DATA_TYPE>(5.0, 50.0), engine, opt_n, h_stock_price);
  sycl::event event_2 = mkl_rng::generate(
      mkl_rng::uniform<DATA_TYPE>(10.0, 25.0), engine, opt_n, h_option_strike);
  sycl::event event_3 = mkl_rng::generate(
      mkl_rng::uniform<DATA_TYPE>(1.0, 5.0), engine, opt_n, h_option_years);
  sycl::event::wait({event_1, event_2, event_3});
}

Binomial::~Binomial() {
  sycl::free(h_call_result, *binomial_queue);
  sycl::free(h_stock_price, *binomial_queue);
  sycl::free(h_option_strike, *binomial_queue);
  sycl::free(h_option_years, *binomial_queue);

  delete binomial_queue;
}

void Binomial::body() {
  constexpr int block_size = num_steps / wg_size;
  static_assert(block_size * wg_size == num_steps);

  // "this" can not be captured to the kernel. So, we need to copy internals of
  // the class to local variables
  DATA_TYPE* h_stock_price_local = this->h_stock_price;
  DATA_TYPE* h_option_years_local = this->h_option_years;
  DATA_TYPE* h_option_strike_local = this->h_option_strike;
  DATA_TYPE* h_call_result_local = this->h_call_result;

  binomial_queue->submit([&](sycl::handler& h) {
    sycl::local_accessor<DATA_TYPE> slm_call{wg_size + 1, h};

    h.template parallel_for(
        sycl::nd_range(sycl::range<1>(opt_n * wg_size),
                       sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> item)
            [[intel::kernel_args_restrict]] [[intel::reqd_sub_group_size(
                sg_size)]] {
              const size_t opt = item.get_global_id(0) / wg_size;
              const DATA_TYPE sx = h_stock_price_local[opt];
              const DATA_TYPE xx = h_option_strike_local[opt];
              const DATA_TYPE tx = h_option_years_local[opt];
              const DATA_TYPE dt = tx / static_cast<DATA_TYPE>(num_steps);
              const DATA_TYPE v_dt = volatility * sycl::sqrt(dt);
              const DATA_TYPE r_dt = risk_free * dt;
              const DATA_TYPE i_f = sycl::exp(r_dt);
              const DATA_TYPE df = sycl::exp(-r_dt);
              const DATA_TYPE u = sycl::exp(v_dt);
              const DATA_TYPE d = sycl::exp(-v_dt);
              const DATA_TYPE pu = (i_f - d) / (u - d);
              const DATA_TYPE pd = static_cast<DATA_TYPE>(1.0) - pu;
              const DATA_TYPE pu_df = pu * df;
              const DATA_TYPE pd_df = pd * df;
              const DATA_TYPE mul_c = v_dt * static_cast<DATA_TYPE>(2.0);
              DATA_TYPE id = v_dt * static_cast<DATA_TYPE>(-num_steps);

              DATA_TYPE local_call[block_size + 1];
              auto wg = item.get_group();
              int local_id = wg.get_local_id(0);
              int block_start = block_size * local_id;
              id += block_start * mul_c;
              for (int i = 0; i < block_size; i++) {
                auto d = sx * sycl::exp(id) - xx;
                local_call[i] = (d > 0) ? d : 0;
                id += mul_c;
              }

              // Handling num_steps step by last item and putting it direclty to
              // SLM last element
              if (local_id == wg_size - 1) {
                auto d = sx * sycl::exp(id) - xx;
                slm_call[wg_size] = (d > 0) ? d : 0;
              }

              // Start at the final tree time step nodes(leaves) and walk
              // backwards to calculate the call option price.
              for (int i = num_steps; i > 0; i--) {
                // Give and get "next block's local_call[j+1]" (local_call[0] in
                // next block) elements across work items
                slm_call[local_id] = local_call[0];
                if (wg_size > sg_size) {
                  item.barrier(sycl::access::fence_space::local_space);
                }
                local_call[block_size] = slm_call[local_id + 1];
                if (wg_size > sg_size) {
                  item.barrier(sycl::access::fence_space::local_space);
                }
                if (block_start <= i) {
                  for (int j = 0; j < block_size; j++) {
                    local_call[j] =
                        pu_df * local_call[j + 1] + pd_df * local_call[j];
                  }
                }
              }
              if (local_id == 0) {
                h_call_result_local[opt] = local_call[0];
              }
            });
  });

  binomial_queue->wait();
}

void Binomial::run() {
  std::printf(
      "%s Precision Binomial Option Pricing version %d.%d running on %s using "
      "DPC++, workgroup size %d, sub-group size %d.\n",
      sizeof(DATA_TYPE) > 4 ? "Double" : "Single", MAJOR, MINOR,
      binomial_queue->get_device().get_info<sycl::info::device::name>().c_str(),
      wg_size, sg_size);

  std::printf("Compiler Version: %s, LLVM %d.%d based.\n", __VERSION__,
              __clang_major__, __clang_minor__);
  std::printf("Driver Version  : %s\n",
              binomial_queue->get_device()
                  .get_info<sycl::info::device::driver_version>()
                  .c_str());
  std::printf("Build Time      : %s %s\n", __DATE__, __TIME__);
  std::printf("Input Dataset   : %d\n", opt_n);
  std::printf("Pricing %d Options with time step of %d.\n", opt_n, num_steps);
  fflush(stdout);
  std::printf("Cold iteration.\n");
  fflush(stdout);
  timer t{};
  t.start();
  body();
  t.stop();
#if REPORT_COLD
  std::printf("Completed in %10.5f seconds. Options per second: %10.5f\n",
              t.duration(), static_cast<double>(opt_n) / (t.duration()));
#endif
#if REPORT_WARM
  std::printf("Warm iteration.\n");
  fflush(stdout);
  t.start();
  body();
  t.stop();
  std::printf("Completed in %10.5f seconds. Options per second: %10.5f\n",
              t.duration(), static_cast<double>(opt_n) / (t.duration()));
#endif
  std::printf("Time Elapsed =  %10.5f seconds\n", t.duration());
  fflush(stdout);
}
