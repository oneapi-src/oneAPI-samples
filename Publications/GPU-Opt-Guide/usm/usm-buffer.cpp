//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iomanip>

#include <CL/sycl.hpp>

#include "utils.hpp"

const int data_size = 10000000;
const int time_steps = 10;
const int device_steps = 1000;

sycl::property_list q_prop{sycl::property::queue::in_order()};
sycl::queue q(q_prop);

float ref_data[data_size];

// Accept a pointer or a host accessor
template <typename T> static void check(T data) {
  return;
  for (int i = 0; i < data_size; i++) {
    if (data[i] != ref_data[i]) {
      std::cout << "Expected: " << ref_data[i] << " got " << data[i] << "\n";
      exit(1);
    }
  }
}

// Accept a pointer or a host accessor
template <typename T> static void init(T data) {
  for (int i = 0; i < data_size; i++)
    data[i] = 0;
}

void put_elapsed_time(timer it) {
  std::cout << std::setw(7) << std::fixed << std::setprecision(4)
            << it.elapsed() << std::endl;
}

static void reference(int stride) {
  init(ref_data);

  // compute results
  for (int i = 0; i < time_steps; i++) {
    // Device
    for (int j = 0; j < device_steps; j++) {
      for (int k = 0; k < data_size; k++) {
        ref_data[k] += 1.0;
      }
    }
    // host
    for (int k = 0; k < data_size; k += stride) {
      ref_data[k] += 1.0;
    }
  }
}

void buffer_data(int stride) {
  // Allocate buffer, initialize on host
  sycl::buffer<float> buffer_data{data_size};
  init(sycl::host_accessor(buffer_data, sycl::write_only, sycl::no_init));

  timer it;
  for (int i = 0; i < time_steps; i++) {

    // Compute on device
    q.submit([&](auto &h) {
      sycl::accessor device_data(buffer_data, h);

      auto compute = [=](auto id) {
        for (int k = 0; k < device_steps; k++)
          device_data[id] += 1.0;
      };
      h.parallel_for(data_size, compute);
    });

    // Compute on host
    sycl::host_accessor host_data(buffer_data);
    for (int i = 0; i < data_size; i += stride)
      host_data[i] += 1.0;
  }
  put_elapsed_time(it);

  const sycl::host_accessor h(buffer_data);
  check(h);
} // buffer_data

void device_usm_data(int stride) {
  // Allocate and initialize host data
  float *host_data = new float[data_size];
  init(host_data);

  // Allocate device data
  float *device_data = sycl::malloc_device<float>(data_size, q);

  timer it;

  for (int i = 0; i < time_steps; i++) {
    // Copy data to device and compute
    q.memcpy(device_data, host_data, sizeof(float) * data_size);
    auto compute = [=](auto id) {
      for (int k = 0; k < device_steps; k++)
        device_data[id] += 1.0;
    };
    q.parallel_for(data_size, compute);

    // Copy data to host and compute
    q.memcpy(host_data, device_data, sizeof(float) * data_size).wait();
    for (int k = 0; k < data_size; k += stride)
      host_data[k] += 1.0;
  }
  q.wait();
  put_elapsed_time(it);

  check(host_data);

  sycl::free(device_data, q);
  delete[] host_data;
} // device_usm_data

void shared_usm_data(int stride) {
  float *data = sycl::malloc_shared<float>(data_size, q);
  init(data);

  timer it;

  for (int i = 0; i < time_steps; i++) {
    auto compute = [=](auto id) {
      for (int k = 0; k < device_steps; k++)
        data[id] += 1.0;
    };
    q.parallel_for(data_size, compute).wait();

    for (int k = 0; k < data_size; k += stride)
      data[k] += 1.0;
  }
  q.wait();
  put_elapsed_time(it);

  check(data);

  sycl::free(data, q);
} // shared_usm_data

void serial(int stride) {
  // Allocate and initialize data
  float *data = new float[data_size];
  init(data);

  timer it;

  for (int i = 0; i < time_steps; i++) {
    for (int j = 0; j < data_size; j++) {
      for (int k = 0; k < device_steps; k++)
        data[j] += 1.0;
    }

    for (int j = 0; j < data_size; j += stride)
      data[j] += 1.0;
  }
  put_elapsed_time(it);

  check(data);

  delete[] data;
} // serial

int main() {
  std::vector strides = {1, 200000};

  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << "\n";
  for (auto stride : strides) {
    std::cout << "stride: " << stride << "\n";
    reference(stride);
    std::cout << "  usm malloc_shared: ";
    shared_usm_data(stride);
    std::cout << "  usm malloc_device: ";
    device_usm_data(stride);
    std::cout << "  buffer:            ";
    buffer_data(stride);
    std::cout << "  serial:            ";
    serial(stride);
  }
}
