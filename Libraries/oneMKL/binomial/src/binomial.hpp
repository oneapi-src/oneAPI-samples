//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef __Binomial_HPP__
#define __Binomial_HPP__

#include <chrono>

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif

#ifndef VERBOSE
#define VERBOSE 0
#endif

/******* VERSION *******/

#define MAJOR 1
#define MINOR 8

/******* VERSION *******/

constexpr float volatility = 0.10f;
constexpr float risk_free = 0.06f;

constexpr int num_steps = 2048;
constexpr int opt_n =
#if SMALL_OPT_N
    480;
#else
    8 * 1024 * 1024;
#endif

#ifndef __clang_major__
#define __clang_major__ 0
#endif
#ifndef __clang_minor__
#define __clang_minor__ 0
#endif
#ifndef __VERSION__
#define __VERSION__ __clang_major__
#endif

class Binomial {
 public:
  Binomial();
  ~Binomial();

  void run();
  void check();

 private:
  DATA_TYPE* h_call_result;
  DATA_TYPE* h_stock_price;
  DATA_TYPE* h_option_strike;
  DATA_TYPE* h_option_years;

  void body();
};

class timer {
 public:
  timer() { start(); }
  void start() { t1_ = std::chrono::steady_clock::now(); }
  void stop() { t2_ = std::chrono::steady_clock::now(); }
  auto duration() { return std::chrono::duration<double>(t2_ - t1_).count(); }

 private:
  std::chrono::steady_clock::time_point t1_, t2_;
};

#endif  // __Binomial_HPP__
