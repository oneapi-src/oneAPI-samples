//==============================================================
// Copyright © 2025 Codeplay Software
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <sycl/sycl.hpp>

#include <iostream>
#include <string>

inline void ensure_full_aspects_support(const sycl::device &dev) {
  std::string error_msg;

  if (!dev.has(sycl::aspect::ext_oneapi_graph)) {
    error_msg += "Error: The device does NOT support ext_oneapi_graph. \n";
  }
  if (!dev.has(sycl::aspect::ext_oneapi_limited_graph)) {
    error_msg +=
        "Error: The device does NOT support ext_oneapi_limited_graph. \n";
  }
  if (!dev.has(sycl::aspect::usm_shared_allocations)) {
    error_msg +=
        "Error: The device does NOT support usm_shared_allocations. \n";
  }

  if (!error_msg.empty()) {
    std::cerr << error_msg;
    std::exit(1);
  }
};

inline void ensure_full_graph_support(const sycl::device &dev) {
  if (!dev.has(sycl::aspect::ext_oneapi_graph)) {
    std::cerr << "Error: The device does NOT support ext_oneapi_graph.\n";
    std::exit(1);
  }
};

inline void ensure_graph_support(const sycl::device &dev) {
  if (!dev.has(sycl::aspect::ext_oneapi_limited_graph)) {
    std::cerr
        << "Error: The device does NOT support ext_oneapi_limited_graph.\n";
    std::exit(1);
  }
};
