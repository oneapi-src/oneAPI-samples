//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

#include <CL/sycl.hpp>
#include <iostream>

namespace devicemanager {

// Report all available SYCL devices
inline void GetDevices() {
  std::cout << "Available devices: \n";
  for (const auto &device : sycl::device::get_devices()) {
    switch (device.get_info<sycl::info::device::device_type>()) {
      case sycl::info::device_type::cpu:
        std::cout << "   CPU:  " << device.get_info<sycl::info::device::name>() << "\n";
        break;
      case sycl::info::device_type::gpu:
        std::cout << "   GPU:  " << device.get_info<sycl::info::device::name>() << "\n";
        break;
      case sycl::info::device_type::host:
        std::cout << "   Host (single-threaded CPU)\n";
        break;
      case sycl::info::device_type::accelerator:
        std::cout << "   Accelerator (not supported): " << device.get_info<sycl::info::device::name>() << "\n";
        break;
      default:
        std::cout << "   Unknown (not supported): " << device.get_info<sycl::info::device::name>() << "\n";
        break;
    }
  }
}

// Singleton DeviceManager
// Ensures consistent use of same device and queue among
// all kernels and subroutines
class DeviceManager {
 public:
  // get the currently active device
  sycl::device &GetCurrentDevice() { return current_device_; }

  // get the currently used device queue
  sycl::queue &GetCurrentQueue() { return current_queue_; }

  // select a new device and queue
  // @return true on success, false otherwise
  bool SelectDevice(const sycl::info::device_type &device_type) {
    // Currently we only support the SYCL Host device, or SYCL CPU device
    for (const auto &device : sycl::device::get_devices()) {
      if (device.get_info<sycl::info::device::device_type>() == device_type) {
        current_device_ = device;
      }
    }

    if (current_device_.get_info<sycl::info::device::device_type>() != device_type) {
      std::cout << "Requested device not available \n";
      GetDevices();
      return false;
    } else {
      if (current_device_.is_host()) {
        std::cout << "Using Host device (single-threaded CPU)\n";
      } else {
        std::cout << "Using " << current_device_.get_info<sycl::info::device::name>() << "\n";
      }

      current_queue_ = sycl::queue(current_device_, sycl::property::queue::in_order());

      return true;
    }
  }

  // Returns the instance of device manager singleton.
  static DeviceManager &instance() {
    static DeviceManager device_manager;
    return device_manager;
  }

  // DeviceManager is a singleton
  // remove all constructors
  DeviceManager(const DeviceManager &) = delete;
  DeviceManager &operator=(const DeviceManager &) = delete;
  DeviceManager(DeviceManager &&) = delete;
  DeviceManager &operator=(DeviceManager &&) = delete;

 private:
  DeviceManager() { current_device_ = sycl::device(sycl::default_selector{}); }

  sycl::device current_device_;
  sycl::queue current_queue_;
};

// Get current queue for current device
inline sycl::queue &GetCurrentQueue() { return DeviceManager::instance().GetCurrentQueue(); }

// Get current device
inline sycl::device &GetCurrentDevice() { return DeviceManager::instance().GetCurrentDevice(); }

// Select a different device
inline bool SelectDevice(const sycl::info::device_type &device_type) {
  return DeviceManager::instance().SelectDevice(device_type);
}
}  // namespace devicemanager
