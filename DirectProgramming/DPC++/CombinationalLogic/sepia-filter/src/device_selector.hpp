//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#ifndef DEVICESELECTOR_HPP
#define DEVICESELECTOR_HPP
#include <cstring>
#include <iostream>
#include <string>
#include "CL/sycl.hpp"

// This is the class provided to SYCL runtime by the application to decide
// on which device to run, or whether to run at all.
// When selecting a device, SYCL runtime first takes (1) a selector provided by
// the program or a default one and (2) the set of all available devices. Then
// it passes each device to the '()' operator of the selector. Device, for
// which '()' returned the highest number, is selected. If a negative number
// was returned for all devices, then the selection process will cause an
// exception.
class MyDeviceSelector : public cl::sycl::device_selector {
 public:
  MyDeviceSelector() {}

  // This is the function which gives a "rating" to devices.
  virtual int operator()(const cl::sycl::device &device) const override {
    // The template parameter to device.get_info can be a variety of properties
    // defined by the SYCL spec's cl::sycl::info:: enum. Properties may have
    // different types. Here we query name which is a string.
    const std::string name = device.get_info<cl::sycl::info::device::name>();

    //uncomment to see the list of devices available on the system
    //std::cout << "Trying device: " << name << "..." << "\n";
    //std::cout << "  Vendor: "
      //        << device.get_info<cl::sycl::info::device::vendor>() << "\n";

    if (device.is_gpu()) return 500;
    if (device.is_accelerator()) return 400;
    if (device.is_cpu()) return 300;
    if (device.is_host()) return 100;
    return -1;
  }
};
#endif
