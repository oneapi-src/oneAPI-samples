//==============================================================
// Copyright © 2020 Intel Corporation
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
class MyDeviceSelector : public sycl::device_selector {
 public:
  MyDeviceSelector(const std::string &p) : pattern(p) {
    // std::cout << "Looking for \"" << p << "\" devices" << std::endl;
  }

  // This is the function which gives a "rating" to devices.
  virtual int operator()(const sycl::device &device) const override {
    // The template parameter to device.get_info can be a variety of properties
    // defined by the SYCL spec's sycl::info:: enum. Properties may have
    // different types. Here we query name which is a string.
    const std::string name = device.get_info<sycl::info::device::name>();
    // std::cout << "Trying device: " << name << "..." << std::endl;
    // std::cout << "  Vendor: " <<
    // device.get_info<sycl::info::device::vendor>() << std::endl;

    // Device with pattern in the name is prioritized:
    return (name.find(pattern) != std::string::npos) ? 100 : 1;
  }

 private:
  std::string pattern;
};

#endif
