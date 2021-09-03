// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

// Our simple asynchronous handler function
auto handle_async_error = [](exception_list elist) {
  for (auto &e : elist) {
    try{ std::rethrow_exception(e); }
    catch ( sycl::exception& e ) {
      std::cout << "ASYNC EXCEPTION!!\n";
      std::cout << e.what() << "\n";
    }
  }
};

void say_device (const queue& Q) {
  std::cout << "Device : " 
    << Q.get_device().get_info<info::device::name>() << "\n";
}

int main() { 
  queue Q1{ gpu_selector{}, handle_async_error };
  queue Q2{ cpu_selector{}, handle_async_error };
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

