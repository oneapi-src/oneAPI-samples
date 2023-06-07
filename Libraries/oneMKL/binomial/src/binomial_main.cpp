//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "binomial.hpp"

// Black-Scholes Reference Implementation
void BlackScholesRefImpl(double& callResult,
                         double Sf,  // Stock price
                         double Xf,  // Option strike
                         double Tf,  // Option years
                         double Rf,  // Riskless rate
                         double Vf   // Volatility rate
) {
  // BSM Formula:
  // https://www.nobelprize.org/prizes/economic-sciences/1997/press-release/
  double S = Sf, L = Xf, t = Tf, r = Rf, sigma = Vf;
  double N_d1 =
      1. / 2. + 1. / 2. *
                    std::erf(((log(S / L) + (r + 0.5 * sigma * sigma) * t) /
                              (sigma * std::sqrt(t))) /
                             std::sqrt(2.));
  double N_d2 =
      1. / 2. + 1. / 2. *
                    std::erf(((log(S / L) + (r - 0.5 * sigma * sigma) * t) /
                              (sigma * std::sqrt(t))) /
                             std::sqrt(2.));
  callResult = (S * N_d1 - L * std::exp(-r * t) * N_d2);
}

void Binomial::check() {
  if (VERBOSE) {
    std::printf("Creating the reference result...\n");
    std::vector<double> h_call_result_host(opt_n);

    for (int opt = 0; opt < opt_n; opt++)
      BlackScholesRefImpl(h_call_result_host[opt], h_stock_price[opt],
                          h_option_strike[opt], h_option_years[opt], risk_free,
                          volatility);

    double sum_delta = 0.0, sum_ref = 0.0, max_delta = 0.0, errorVal = 0.0;

    for (int i = 0; i < opt_n; i++) {
      double ref = h_call_result_host[i];
      auto delta = std::fabs(ref - h_call_result[i]);
      if (delta > max_delta) {
        max_delta = delta;
      }
      sum_delta += delta;
      sum_ref += std::fabs(ref);
    }
    if (sum_ref > 1E-5)
      std::printf("L1 norm: %E\n", errorVal = sum_delta / sum_ref);
    else
      std::printf("Avg. diff: %E\n", errorVal = sum_delta / opt_n);
    std::printf((errorVal < 5e-4) ? "TEST PASSED\n" : "TEST FAILED\n");
  }
}

int main(int argc, char** argv) {
  Binomial test;
  test.run();
  test.check();
  return 0;
}
