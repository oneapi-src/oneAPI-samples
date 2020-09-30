//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// This is a simple DPC++ program that accompanies the Getting Started
// Guide of the debugger.  The kernel does not compute anything
// particularly interesting; it is designed to illustrate the most
// essential features of the debugger when the target device is CPU or
// GPU.

#include <CL/sycl.hpp>
#include <iostream>
// Location of file: <oneapi-root>/dev-utilities/<version>/include
#include "dpc_common.hpp"
#include "selector.hpp"

using namespace std;
using namespace sycl;

// A device function, called from inside the kernel.
static size_t GetDim(id<1> wi, int dim) {
  return wi[dim];
}

int main(int argc, char *argv[]) {
  constexpr size_t length = 64;
  int input[length];
  int output[length];

  // Initialize the input
  for (int i = 0; i < length; i++)
    input[i] = i + 100;

  try {
    CustomSelector selector(GetDeviceType(argc, argv));
    queue q(selector, dpc_common::exception_handler);
    cout << "[SYCL] Using device: ["
         << q.get_device().get_info<info::device::name>()
         << "] from ["
         << q.get_device().get_platform().get_info<info::platform::name>()
         << "]\n";

    range data_range{length};
    buffer buffer_in{input, data_range};
    buffer buffer_out{output, data_range};

    q.submit([&](auto &h) {
      accessor in(buffer_in, h, read_only);
      accessor out(buffer_out, h, write_only);

      // kernel-start
      h.parallel_for(data_range, [=](id<1> index) {
        size_t id0 = GetDim(index, 0);
        int element = in[index];  // breakpoint-here
        int result = element + 50;
        if (id0 % 2 == 0) {
          result = result + 50;  // then-branch
        } else {
          result = -1;  // else-branch
        }
        out[index] = result;
      });
      // kernel-end
    });

    q.wait_and_throw();
  } catch (sycl::exception const& e) {
    cout << "fail; synchronous exception occurred: " << e.what() << "\n";
    return -1;
  }

  // Verify the output
  for (int i = 0; i < length; i++) {
    int result = (i % 2 == 0) ? (input[i] + 100) : -1;
    if (output[i] != result) {
      cout << "fail; element " << i << " is " << output[i] << "\n";
      return -1;
    }
  }

  cout << "success; result is correct.\n";
  return 0;
}
