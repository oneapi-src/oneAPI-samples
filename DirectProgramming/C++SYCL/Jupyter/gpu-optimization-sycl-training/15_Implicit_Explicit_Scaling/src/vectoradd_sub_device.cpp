//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

void kernel_compute_vadd(sycl::queue &q, float *a, float *b, float *c, size_t n) {
  q.parallel_for(n, [=](auto i) {
    c[i] = a[i] + b[i];
  });
}

int main() {
  const int N = 1680;

  // Define 3 arrays
  float *a = static_cast<float *>(malloc(N * sizeof(float)));
  float *b = static_cast<float *>(malloc(N * sizeof(float)));
  float *c = static_cast<float *>(malloc(N * sizeof(float)));

  // Initialize matrices with values
  for (int i = 0; i < N; i++){
    a[i] = 1;
    b[i] = 2;
    c[i] = 0;
  }

  sycl::queue q_root;
  sycl::device RootDevice =  q_root.get_device();
  std::cout << "Device: " << RootDevice.get_info<sycl::info::device::name>() << "\n";
  std::cout << "-EUs  : " << RootDevice.get_info<sycl::info::device::max_compute_units>() << "\n\n";

  //# Check if GPU can be partitioned (stacks/Stack)
  std::vector<sycl::queue> q;
  auto num_devices = RootDevice.get_info<sycl::info::device::partition_max_sub_devices>();
  if(num_devices > 0){
    std::cout << "-partition_max_sub_devices: " << num_devices << "\n\n";
    std::vector<sycl::device> SubDevices = RootDevice.create_sub_devices<
                  sycl::info::partition_property::partition_by_affinity_domain>(
                                                  sycl::info::partition_affinity_domain::numa);
    for (auto &SubDevice : SubDevices) {
      q.push_back(sycl::queue(SubDevice));
      std::cout << "Sub-Device: " << SubDevice.get_info<sycl::info::device::name>() << "\n";
      std::cout << "-EUs      : " << SubDevice.get_info<sycl::info::device::max_compute_units>() << "\n";
    }  
  } else {
     std::cout << "-cannot partition to sub-device, running on root-device " << "\n\n";
     num_devices = 1;
     q.push_back(q_root);
  }


  // device mem alloc for vectors a,b,c for each device
  float *da[num_devices];
  float *db[num_devices];
  float *dc[num_devices];
  for (int i = 0; i < num_devices; i++) {
    da[i] = sycl::malloc_device<float>(N/num_devices, q[i]);
    db[i] = sycl::malloc_device<float>(N/num_devices, q[i]);
    dc[i] = sycl::malloc_device<float>(N/num_devices, q[i]);
  }

  // memcpy for matrix and b to device alloc
  for (int i = 0; i < num_devices; i++) {
    q[i].memcpy(&da[i][0], &a[i*N/num_devices], N/num_devices * sizeof(float));
    q[i].memcpy(&db[i][0], &b[i*N/num_devices], N/num_devices * sizeof(float));
  }

  // wait for copy to complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // submit vector-add kernels to all devices
  for (int i = 0; i < num_devices; i++)
    kernel_compute_vadd(q[i], da[i], db[i], dc[i], N/num_devices);

  // wait for compute complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // copy back result to host
  for (int i = 0; i < num_devices; i++)
    q[i].memcpy(&c[i*N/num_devices], &dc[i][0], N/num_devices * sizeof(float));

  // wait for copy to complete
  for (int i = 0; i < num_devices; i++)
    q[i].wait();

  // print output
  for (int i = 0; i < N; i++) std::cout << c[i] << " ";
  std::cout << "\n";

  free(a);
  free(b);
  free(c);
  for (int i = 0; i < num_devices; i++) {
    sycl::free(da[i], q[i]);
    sycl::free(db[i], q[i]);
    sycl::free(dc[i], q[i]);
  }
  return 0;
}
