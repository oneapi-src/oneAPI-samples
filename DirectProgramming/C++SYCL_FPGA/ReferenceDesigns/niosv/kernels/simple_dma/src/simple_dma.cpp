//  Copyright (c) 2022 Intel Corporation
//  SPDX-License-Identifier: MIT

#include <stdlib.h>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

// define buffer locations so the IP can have two unique Avalon memory-mapped
// host interfaces
static constexpr int kBL1 = 1;
static constexpr int kBL2 = 2;

// define alignment of Avalon memory-mapped host interfaces to be 4 bytes per
// read
static constexpr int kAlignment = 4;

struct SimpleDMA {
  using params1 = decltype(sycl::ext::oneapi::experimental::properties(
      // give this a unique Avalon memory-mapped host interface
      sycl::ext::intel::experimental::buffer_location<kBL1>,
      sycl::ext::intel::experimental::dwidth<32>,
      sycl::ext::intel::experimental::awidth<32>,
      sycl::ext::intel::experimental::maxburst<15>,
      sycl::ext::oneapi::experimental::alignment<kAlignment>,
      // latency (choose 0-latency so that waitrequest will work)
      sycl::ext::intel::experimental::latency<0>,
      sycl::ext::intel::experimental::read_write_mode_read));

  using params2 = decltype(sycl::ext::oneapi::experimental::properties(
      // give this a unique Avalon memory-mapped host interface
      sycl::ext::intel::experimental::buffer_location<kBL2>,
      sycl::ext::intel::experimental::dwidth<32>,
      sycl::ext::intel::experimental::awidth<32>,
      sycl::ext::intel::experimental::maxburst<15>,
      sycl::ext::oneapi::experimental::alignment<kAlignment>,
      // latency (choose 0-latency so that waitrequest will work)
      sycl::ext::intel::experimental::latency<0>,
      sycl::ext::intel::experimental::read_write_mode_write));

  // Struct members will be interpreted as kernel arguments. The pointers are
  // declared first since they are 64-bit types and won't get split up
  sycl::ext::oneapi::experimental::annotated_arg<unsigned int *, params1>
      source;
  sycl::ext::oneapi::experimental::annotated_arg<unsigned int *, params2> dest;

  // measured in bytes, must be a multiple of 4. This is only 32 bits wide so if
  // it was declared first it would result in the pointers getting split across
  // multiple CSR lines.
  unsigned int length_bytes;

  // This accelerator will be controlled via its Avalon-MM agent interface, so
  // no kernel properties are set.

  // Implementation of the DMA kernel.
  void operator()() const {
    // This loop does not handle partial accesses (less than 4 bytes) at the
    // start and end of the source/destination so ensure they are at least
    // 4-byte aligned
    for (unsigned int i = 0; i < (length_bytes / 4); i++) {
      dest[i] = source[i];
    }
  }
};

constexpr int kLen = 128;  // bytes

int main() {
  // Use compile-time macros to select either:
  //  - the FPGA emulator device (CPU emulation of the FPGA)
  //  - the FPGA device (a real FPGA)
  //  - the simulator device
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  // create the device queue
  sycl::queue q(selector, fpga_tools::exception_handler);

  // make sure the device supports USM host allocations
  auto device = q.get_device();

  std::cout << "Running on device: "
            << device.get_info<sycl::info::device::name>().c_str() << std::endl;

  unsigned int *src = sycl::aligned_alloc_shared<unsigned int>(
      kAlignment, kLen, q,
      sycl::ext::intel::experimental::property::usm::buffer_location(kBL1));
  unsigned int *dest = sycl::aligned_alloc_shared<unsigned int>(
      kAlignment, kLen, q,
      sycl::ext::intel::experimental::property::usm::buffer_location(kBL2));
  unsigned int len = kLen;

  // ensure that shared pointers are successfully allocated
  if (nullptr == src) {
    std::cerr << "failed to allocate pointer src: make sure that awidth is "
                 "sufficient for the allocation size."
              << std::endl;
    return EXIT_FAILURE;
  }
  if (nullptr == dest) {
    std::cerr << "failed to allocate pointer dest: make sure that awidth is "
                 "sufficient for the allocation size."
              << std::endl;
    return EXIT_FAILURE;
  }

  // pre-load
  for (int i = 0; i < len; i++) {
    src[i] = len - i;
    dest[i] = 0;
  }

  // line below is what associates the name "SimpleDMA" to the kernel
  q.single_task(SimpleDMA{src, dest, len}).wait();

  // check results
  bool passed = true;
  for (int i = 0; i < (len / 4); i++) {
    bool ok = (src[i] == dest[i]);
    passed &= ok;

    if (!passed) {
      std::cerr << "ERROR: [" << i << "] expected " << src[i] << " saw "
                << dest[i] << ". " << std::endl;
    }
  }

  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
  sycl::free(src, q);
  sycl::free(dest, q);

  return EXIT_SUCCESS;
}
