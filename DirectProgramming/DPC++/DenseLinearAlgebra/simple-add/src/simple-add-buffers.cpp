//==============================================================
// Iota is the equivalent of a Hello, World! sample for data parallel programs.
// Building and running the sample verifies that your development environment
// is setup correctly and demonstrates the use of the core features of DPC++.
// This sample runs on both CPU and GPU (or FPGA). When run, it computes on both
// the CPU and offload device, then compares results. If the code executes on
// both CPU and the offload device, the name of the offload device and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// DPC++ material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <array>
#include <iostream>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

#if FPGA || FPGA_EMULATOR
// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif
#endif

using namespace sycl;
using namespace std;

// Array type and data size for this example.
constexpr size_t array_size = 10000;
typedef array<int, array_size> IntArray;

//************************************
// Iota in DPC++ on device.
//************************************
void IotaParallel(queue &q, IntArray &a_array, int value) {
  // Create the range object for the array managed by the buffer.
  range num_items{a_array.size()};

  // Create buffer that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer a_buf(a_array);

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](auto &h) {
    // Create an accessor with write permission.
    accessor a(a_buf, h, write_only);

    // Use parallel_for to populate consecutive numbers starting with a
    // specified value in parallel on device. This executes the kernel.
    //    1st parameter is the number of work items to use.
    //    2nd parameter is the kernel, a lambda that specifies what to do per
    //    work item. The parameter of the lambda is the work item id.
    // DPC++ supports unnamed lambda kernel by default.
    h.parallel_for(num_items, [=](auto i) { a[i] = value + i; });
  });
}

//************************************
// Demonstrate iota both sequential on CPU and parallel on device.
//************************************
int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  INTEL::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  INTEL::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  // Create array objects with "array_size" to store data.
  IntArray sequential, parallel;
  constexpr int value = 100000;

  // Sequential iota.
  for (size_t i = 0; i < sequential.size(); i++) sequential[i] = value + i;

  try {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    cout << "Running on device: "
         << q.get_device().get_info<info::device::name>() << "\n";
    cout << "Array size: " << parallel.size() << "\n";

    // Parallel iota in DPC++.
    IotaParallel(q, parallel, value);
  } catch (std::exception const &e) {
    cout << "An exception is caught while computing on device.\n";
    terminate();
  }

  // Verify two results are equal.
  for (size_t i = 0; i < sequential.size(); i++) {
    if (parallel[i] != sequential[i]) {
      cout << "Failed on device.\n";
      return -1;
    }
  }

  int indices[]{0, 1, 2, (sequential.size() - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  // Print out iota result.
  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) cout << "...\n";
    cout << "[" << j << "]: " << j << " + " << value << " = "
         << parallel[j] << "\n";
  }

  cout << "Successfully completed on device.\n";
  return 0;
}
