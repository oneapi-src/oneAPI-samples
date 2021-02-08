/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 * Copyright (c) 2019-2021 Intel Corporation (oneAPI modifications)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <CL/sycl.hpp>
#include <cstdio>
#include <dpct/dpct.hpp>

// using MACRO to allocate memory inside kernel
#define NUM_3D_BOX_CORNERS_MACRO 8
#define NUM_2D_BOX_CORNERS_MACRO 4

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

inline void getSyclDevices() {
  std::cout << "Available devices: \n";
  for (uint i = 0; i < dpct::dev_mgr::instance().device_count(); i++) {
    auto &device = dpct::dev_mgr::instance().get_device(i);
    switch (device.get_info<cl::sycl::info::device::device_type>()) {
      case cl::sycl::info::device_type::cpu:
        std::cout << "   CPU:  " << device.get_info<cl::sycl::info::device::name>() << "\n";
        break;
      case cl::sycl::info::device_type::gpu:
        std::cout << "   GPU:  " << device.get_info<cl::sycl::info::device::name>() << "\n";
        break;
      case cl::sycl::info::device_type::host:
        std::cout << "   Host (single-threaded CPU)\n";
        break;
      case cl::sycl::info::device_type::accelerator:
        std::cout << "   Accelerator (not supported): " << device.get_info<cl::sycl::info::device::name>() << "\n";
        break;
      default:
        std::cout << "   Unknown (not supported): " << device.get_info<cl::sycl::info::device::name>() << "\n";
        break;
    }
  }
}

inline bool changeDefaultSyclDevice(cl::sycl::info::device_type deviceType) {
  // Currently we only support the SYCL Host device, or SYCL CPU device
  for (uint i = 0; i < dpct::dev_mgr::instance().device_count(); i++) {
    auto &device = dpct::dev_mgr::instance().get_device(i);
    if (device.get_info<cl::sycl::info::device::device_type>() == deviceType) {
      dpct::dev_mgr::instance().select_device(i);
      break;
    }
  }

  if (dpct::get_current_device().get_info<cl::sycl::info::device::device_type>() != deviceType) {
    std::cout << "Requested device not available \n";
    getSyclDevices();
    return false;
  } else {
    if (dpct::get_current_device().is_host()) {
      std::cout << "Using Host device (single-threaded CPU)\n";
    } else {
      std::cout << "Using " << dpct::get_current_device().get_info<cl::sycl::info::device::name>() << "\n";
    }

    return true;
  }
}
