// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;

// BEGIN CODE SNIP

// Our simple asynchronous handler function
auto handle_async_error = [](exception_list elist) {
  for (auto &e : elist) {
    try { std::rethrow_exception(e); }
    catch ( sycl::exception& e ) {
      // Print information about the asynchronous exception
    }
  }

  // Terminate abnormally to make clear to user that something unhandled happened
  std::terminate();
};

// END CODE SNIP


void say_device(const queue& Q) {
  std::cout << "Device : " 
    << Q.get_device().get_info<info::device::name>() << "\n";
}

int main() { 
  queue Q1{ gpu_selector_v, handle_async_error };
  queue Q2{ cpu_selector_v, handle_async_error };
  say_device(Q1);
  say_device(Q2);

  try {
    Q1.submit([&] (handler &h){
        // Empty command group is illegal and generates an error
        },
        Q2); // Secondary/backup queue!
  } catch (...) {}  // Discard regular C++ exceptions for this example
  return 0;
}

