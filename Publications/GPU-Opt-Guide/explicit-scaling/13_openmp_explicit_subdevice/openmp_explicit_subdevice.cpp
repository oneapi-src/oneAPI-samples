//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <assert.h>
#include <iostream>
#include <omp.h>
#include <stdint.h>
#ifndef NUM_SUBDEVICES
#define NUM_SUBDEVICES 1
#endif

#ifndef DEVKIND
#define DEVKIND 0
#endif

template <int num_subdevices> struct mptr {
  float *p[num_subdevices];
};

int main(int argc, char **argv) {
  constexpr int SIZE = 8e6;
  constexpr int SIMD_SIZE = 32;
  constexpr std::size_t TOTAL_SIZE = SIZE * SIMD_SIZE;
  constexpr int num_subdevices = NUM_SUBDEVICES;

  mptr<num_subdevices> device_ptr_a;
  mptr<num_subdevices> device_ptr_b;
  mptr<num_subdevices> device_ptr_c;

  const int device_id = omp_get_default_device();
  std::cout << "device_id = " << device_id << std::endl;

  for (int sdev = 0; sdev < num_subdevices; ++sdev) {
    device_ptr_a.p[sdev] =
        static_cast<float *>(malloc(TOTAL_SIZE * sizeof(float)));
    device_ptr_b.p[sdev] =
        static_cast<float *>(malloc(TOTAL_SIZE * sizeof(float)));
    device_ptr_c.p[sdev] =
        static_cast<float *>(malloc(TOTAL_SIZE * sizeof(float)));

#pragma omp target enter data map(alloc                                        \
                                  : device_ptr_a.p[sdev] [0:TOTAL_SIZE])       \
    device(device_id) subdevice(DEVKIND, sdev)
#pragma omp target enter data map(alloc                                        \
                                  : device_ptr_b.p[sdev] [0:TOTAL_SIZE])       \
    device(device_id) subdevice(DEVKIND, sdev)
#pragma omp target enter data map(alloc                                        \
                                  : device_ptr_c.p[sdev] [0:TOTAL_SIZE])       \
    device(device_id) subdevice(DEVKIND, sdev)
  }
  std::cout << "memory footprint per GPU = "
            << 3 * (std::size_t)(TOTAL_SIZE) * sizeof(float) * 1E-9 << " GB"
            << std::endl;

#pragma omp parallel for
  for (int sdev = 0; sdev < num_subdevices; ++sdev) {
    float *a = device_ptr_a.p[sdev];
    float *b = device_ptr_b.p[sdev];

#pragma omp target teams distribute parallel for device(device_id)             \
    subdevice(LEVEL, sdev)
    for (int i = 0; i < TOTAL_SIZE; ++i) {
      a[i] = i + 0.5;
      b[i] = i - 0.5;
    }
  }

  const int no_max_rep = 200;
  double time = 0.0;
  for (int irep = 0; irep < no_max_rep + 1; ++irep) {
    if (irep == 1)
      time = omp_get_wtime();

#pragma omp parallel for num_threads(num_subdevices)
    for (int sdev = 0; sdev < num_subdevices; ++sdev) {
      float *a = device_ptr_a.p[sdev];
      float *b = device_ptr_b.p[sdev];
      float *c = device_ptr_c.p[sdev];

#pragma omp target teams distribute parallel for device(device_id)             \
    subdevice(LEVEL, sdev)
      for (int i = 0; i < TOTAL_SIZE; ++i) {
        c[i] = a[i] + b[i];
      }
    }
  }

  time = omp_get_wtime() - time;
  time = time / no_max_rep;

  const std::size_t streamed_bytes =
      3 * (std::size_t)(TOTAL_SIZE)*num_subdevices * sizeof(float);
  std::cout << "bandwidth = " << (streamed_bytes / time) * 1E-9 << " GB/s"
            << std::endl;
  std::cout << "time = " << time << " s" << std::endl;
  std::cout.precision(10);

  for (int sdev = 0; sdev < num_subdevices; ++sdev) {
#pragma omp target update from(device_ptr_c.p[sdev][:TOTAL_SIZE])              \
    device(device_id) subdevice(LEVEL, sdev)
    std::cout << "-GPU: device id = : " << sdev << std::endl;
    std::cout << "target result:" << std::endl;
    std::cout << "c[" << 0 << "] = " << device_ptr_c.p[sdev][0] << std::endl;
    std::cout << "c[" << SIMD_SIZE - 1
              << "] = " << device_ptr_c.p[sdev][SIMD_SIZE - 1] << std::endl;
    std::cout << "c[" << TOTAL_SIZE / 2
              << "] = " << device_ptr_c.p[sdev][TOTAL_SIZE / 2] << std::endl;
    std::cout << "c[" << TOTAL_SIZE - 1
              << "] = " << device_ptr_c.p[sdev][TOTAL_SIZE - 1] << std::endl;
  }

  for (int sdev = 0; sdev < num_subdevices; ++sdev) {
    for (int i = 0; i < TOTAL_SIZE; ++i) {
      assert((int)(device_ptr_c.p[sdev][i]) ==
             (int)(device_ptr_c.p[sdev][i] +
                   device_ptr_a.p[sdev][i] * device_ptr_b.p[sdev][i]));
    }
  }

  for (int sdev = 0; sdev < num_subdevices; ++sdev) {
#pragma omp target exit data map(release : device_ptr_a.p[sdev][:TOTAL_SIZE])
#pragma omp target exit data map(release : device_ptr_b.p[sdev][:TOTAL_SIZE])
#pragma omp target exit data map(release : device_ptr_a.p[sdev][:TOTAL_SIZE])
  }
}
