//==============================================================
// Iota is the equivalent of a Hello, World! sample for data parallel programs.
// Building and running the sample verifies that your development environment
// is setup correctly and demonstrates the use of the core features of SYCL.
// This sample runs on both CPU and GPU (or FPGA). When run, it computes on both
// the CPU and offload device, then compares results. If the code executes on
// both CPU and the offload device, the name of the offload device and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue and kernel.
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <array>
#include <iostream>

#if FPGA || FPGA_EMULATOR
  #include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;
using namespace std;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

// Array size for this example.
constexpr size_t array_size = 10000;

//************************************
// Iota in SYCL on device.
//************************************
void IotaParallel(queue &q, int *a, size_t size, int value) {
  // Create the range object for the array.
  range num_items{size};

  // Use parallel_for to populate consecutive numbers starting with a specified
  // value in parallel on device. This executes the kernel.
  //    1st parameter is the number of work items to use.
  //    2nd parameter is the kernel, a lambda that specifies what to do per
  //    work item. The parameter of the lambda is the work item id.
  // SYCL supports unnamed lambda kernel by default.
  auto e = q.parallel_for(num_items, [=](auto i) { a[i] = value + i; });

  // q.parallel_for() is an asynchronous call. SYCL runtime enqueues and runs
  // the kernel asynchronously. Wait for the asynchronous call to complete.
  e.wait();
}

//************************************
// Demonstrate iota both sequential on CPU and parallel on device.
//************************************
int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // Intel extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector selector;
#elif FPGA
  // Intel extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector selector;
#else
  // The default device selector will select the most performant device.
  auto selector{default_selector_v};
#endif

  constexpr int value = 100000;

  try {
    queue q(selector, exception_handler);

    // Print out the device information used for the kernel code.
    cout << "Running on device: "
         << q.get_device().get_info<info::device::name>() << "\n";
    cout << "Array size: " << array_size << "\n";

    int *sequential = malloc_shared<int>(array_size, q);
    int *parallel = malloc_shared<int>(array_size, q);

    if ((sequential == nullptr) || (parallel == nullptr)) {
      if (sequential != nullptr) free(sequential, q);
      if (parallel != nullptr) free(parallel, q);

      cout << "Shared memory allocation failure.\n";
      return -1;
    }

    // Sequential iota.
    for (size_t i = 0; i < array_size; i++) sequential[i] = value + i;

    // Parallel iota in SYCL.
    IotaParallel(q, parallel, array_size, value);

    // Verify two results are equal.
    for (size_t i = 0; i < array_size; i++) {
      if (parallel[i] != sequential[i]) {
        cout << "Failed on device.\n";
        return -1;
      }
    }

    int indices[]{0, 1, 2, (array_size - 1)};
    constexpr size_t indices_size = sizeof(indices) / sizeof(int);

    // Print out iota result.
    for (int i = 0; i < indices_size; i++) {
      int j = indices[i];
      if (i == indices_size - 1) cout << "...\n";
      cout << "[" << j << "]: " << j << " + " << value << " = "
           << sequential[j] << "\n";
    }

    free(sequential, q);
    free(parallel, q);
  } catch (std::exception const &e) {
    cout << "An exception is caught while computing on device.\n";
    terminate();
  }

  cout << "Successfully completed on device.\n";
  return 0;
}
