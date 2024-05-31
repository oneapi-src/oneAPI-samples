//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

int main(){
  int N = 1024000000;

  // get all GPUs devices into a vector
  auto gpus = sycl::platform(sycl::gpu_selector_v).get_devices();

  if(gpus.size() > 1){
    sycl::queue q0(gpus[0], sycl::property::queue::enable_profiling());
    sycl::queue q1(gpus[1], sycl::property::queue::enable_profiling());
    std::cout << "GPU0: " << q0.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "GPU1: " << q1.get_device().get_info<sycl::info::device::name>() << "\n";

	auto host_mem = sycl::malloc_host<int>(N, q0);
    for (int i=0;i<N;i++) host_mem[i] = i;
        
	auto dev0_mem = sycl::malloc_device<int>(N, q0);
	auto dev0_mem2 = sycl::malloc_device<int>(N, q0);
    auto dev1_mem = sycl::malloc_device<int>(N, q1);
        
	// host to GPU0 copy
    auto event_h2d0 = q0.memcpy(dev0_mem, host_mem, N*sizeof(int));
    q0.wait();

    // GPU0 to GPU1 copy q0
    auto event_d2d_q0 = q0.memcpy(dev1_mem, dev0_mem, N*sizeof(int));
    q0.wait();

    // GPU0 to GPU0 copy
    auto event_d2d_same0 = q0.memcpy(dev0_mem2, dev0_mem, N*sizeof(int));
    q0.wait();

    std::cout << host_mem[0] << " ... " << host_mem[N-1] << " (1M int, 3.8GB Tx)\n"; 

    // free allocation
	sycl::free(host_mem, q0);
	sycl::free(dev0_mem, q0);
	sycl::free(dev0_mem2, q0);
    sycl::free(dev1_mem, q1);

    // Print kernel profile times for copy
    auto startK = event_h2d0.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto endK = event_h2d0.get_profiling_info<sycl::info::event_profiling::command_end>();
    std::cout << "Host to GPU0 Copy [q0]: " << (endK - startK) / 1e+9 << " seconds\n";
    startK = event_d2d_q0.get_profiling_info<sycl::info::event_profiling::command_start>();
    endK = event_d2d_q0.get_profiling_info<sycl::info::event_profiling::command_end>();
	std::cout << "GPU0 to GPU1 Copy [q0]: " << (endK - startK) / 1e+9 << " seconds\n";
	startK = event_d2d_same0.get_profiling_info<sycl::info::event_profiling::command_start>();
    endK = event_d2d_same0.get_profiling_info<sycl::info::event_profiling::command_end>();
    std::cout << "GPU0 to GPU0 Copy [q0]: " << (endK - startK) / 1e+9 << " seconds\n";
  } else {
    std::cout << "Multiple GPUs not available\n";
  }
  return 0;
}
