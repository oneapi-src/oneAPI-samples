//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

static constexpr size_t N = 5280; // global size
static constexpr size_t B = 32;   // WG size

void kernel_compute_mm(sycl::queue &q, float *a, float *b, float *c, size_t n,
                       size_t wg) {
  q.parallel_for(
      sycl::nd_range<2>(sycl::range<2>{n, n}, sycl::range<2>{wg, wg}),
      [=](sycl::nd_item<2> item) {
        const int i = item.get_global_id(0);
        const int j = item.get_global_id(1);
        float temp = 0.0f;
        for (int k = 0; k < N; k++) {
          temp += a[i * N + k] * b[k * N + j];
        }
        c[i * N + j] = temp;
      });
}

int main() {
  auto start =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  // find all CCS / Tiles in GPU
  auto device = sycl::device(sycl::gpu_selector_v);
  std::cout << "\nGPU: " << device.get_info<sycl::info::device::name>() << " ("
            << device.get_info<sycl::info::device::max_compute_units>()
            << ")\n";
  std::vector<sycl::device> subdevices;
  std::vector<sycl::device> subsubdevices;
  auto part_prop = device.get_info<sycl::info::device::partition_properties>();
  if (part_prop.empty()) {
    std::cout << "No partition_properties\n";
  } else {
    for (int i = 0; i < part_prop.size(); i++) {
      // Check if device can be partitioned into Tiles
      if (part_prop[i] ==
          sycl::info::partition_property::partition_by_affinity_domain) {
        auto sub_devices = device.create_sub_devices<
            sycl::info::partition_property::partition_by_affinity_domain>(
            sycl::info::partition_affinity_domain::numa);
        for (int j = 0; j < sub_devices.size(); j++) {
          subdevices.push_back(sub_devices[j]);
          std::cout
              << "\nTile" << j << ": "
              << subdevices[j].get_info<sycl::info::device::name>() << " ("
              << subdevices[j].get_info<sycl::info::device::max_compute_units>()
              << ")\n";
          auto part_prop1 =
              subdevices[j]
                  .get_info<sycl::info::device::partition_properties>();
          if (part_prop1.empty()) {
            std::cout << "No partition_properties\n";
          } else {
            for (int i = 0; i < part_prop1.size(); i++) {
              // Check if Tile can be partitioned into Slices (CCS)
              if (part_prop1[i] == sycl::info::partition_property::
                                       ext_intel_partition_by_cslice) {
                auto sub_devices = subdevices[j]
                                       .create_sub_devices<
                                           sycl::info::partition_property::
                                               ext_intel_partition_by_cslice>();
                for (int k = 0; k < sub_devices.size(); k++) {
                  subsubdevices.push_back(sub_devices[k]);
                  std::cout
                      << "Slice" << k << ": "
                      << subsubdevices[k].get_info<sycl::info::device::name>()
                      << " ("
                      << subsubdevices[k]
                             .get_info<sycl::info::device::max_compute_units>()
                      << ")\n";
                }
                break;
              } else {
                std::cout << "No ext_intel_partition_by_cslice\n";
              }
            }
          }
        }
        break;
        // Check if device can be partitioned into Slices (CCS)
      } else if (part_prop[i] == sycl::info::partition_property::
                                     ext_intel_partition_by_cslice) {
        auto sub_devices = device.create_sub_devices<
            sycl::info::partition_property::ext_intel_partition_by_cslice>();
        for (int k = 0; k < sub_devices.size(); k++) {
          subsubdevices.push_back(sub_devices[k]);
          std::cout << "Slice" << k << ": "
                    << subsubdevices[k].get_info<sycl::info::device::name>()
                    << " ("
                    << subsubdevices[k]
                           .get_info<sycl::info::device::max_compute_units>()
                    << ")\n";
        }
        break;
      } else {
        std::cout << "No ext_intel_partition_by_cslice or "
                     "partition_by_affinity_domain\n";
      }
    }
  }

  // Set devices to submit compute kernel
  std::vector<sycl::device> devices(1, device);
  if (subsubdevices.size())
    devices = subsubdevices;
  else if (subdevices.size())
    devices = subdevices;
  auto num_devices = devices.size();

  // Define matrices
  float *matrix_a[num_devices];
  float *matrix_b[num_devices];
  float *matrix_c[num_devices];

  float v1 = 2.f;
  float v2 = 3.f;
  for (int n = 0; n < num_devices; n++) {
    matrix_a[n] = static_cast<float *>(malloc(N * N * sizeof(float)));
    matrix_b[n] = static_cast<float *>(malloc(N * N * sizeof(float)));
    matrix_c[n] = static_cast<float *>(malloc(N * N * sizeof(float)));

    // Initialize matrices with values
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        matrix_a[n][i * N + j] = v1++;
        matrix_b[n][i * N + j] = v2++;
        matrix_c[n][i * N + j] = 0.f;
      }
  }

  float *da[num_devices];
  float *db[num_devices];
  float *dc[num_devices];

  std::vector<sycl::queue> q(num_devices);

  // create queues for each device
  std::cout << "\nSubmitting Compute Kernel to Devices:\n";
  for (int i = 0; i < num_devices; i++) {
    q[i] = sycl::queue(devices[i]);
    std::cout
        << "Device" << i << ": "
        << q[i].get_device().get_info<sycl::info::device::name>() << " ("
        << q[i].get_device().get_info<sycl::info::device::max_compute_units>()
        << ")\n";
  }

  // device mem alloc for matrix a,b,c for each device
  for (int i = 0; i < num_devices; i++) {
    da[i] = sycl::malloc_device<float>(N * N, q[i]);
    db[i] = sycl::malloc_device<float>(N * N, q[i]);
    dc[i] = sycl::malloc_device<float>(N * N, q[i]);
  }

  // warm up: kernel submit with zero size
  for (int i = 0; i < num_devices; i++)
    kernel_compute_mm(q[i], da[i], db[i], dc[i], 0, 0);

  // kernel sync
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // memcpy for matrix and b to device alloc
  for (int i = 0; i < num_devices; i++) {
    q[i].memcpy(&da[i][0], &matrix_a[i][0], N * N * sizeof(float));
    q[i].memcpy(&db[i][0], &matrix_b[i][0], N * N * sizeof(float));
  }

  // wait for copy to complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // submit matrix multiply kernels to all devices
  for (int i = 0; i < num_devices; i++)
    kernel_compute_mm(q[i], da[i], db[i], dc[i], N, B);

  // wait for compute complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // copy back result to host
  for (int i = 0; i < num_devices; i++)
    q[i].memcpy(&matrix_c[i][0], &dc[i][0], N * N * sizeof(float));

  // wait for copy to complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // print first element of result matrix
  std::cout << "\nMatrix Multiplication Complete\n";
  for (int i = 0; i < num_devices; i++)
    std::cout << "device" << i << ": matrix_c[0][0]=" << matrix_c[i][0] << "\n";

  for (int i = 0; i < num_devices; i++) {
    free(matrix_a[i]);
    free(matrix_b[i]);
    free(matrix_c[i]);
    sycl::free(da[i], q[i]);
    sycl::free(db[i], q[i]);
    sycl::free(dc[i], q[i]);
  }

  auto duration =
      std::chrono::high_resolution_clock::now().time_since_epoch().count() -
      start;
  std::cout << "Compute Duration: " << duration / 1e+9 << " seconds\n";
  return 0;
}
