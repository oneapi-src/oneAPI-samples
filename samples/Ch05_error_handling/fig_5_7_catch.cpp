// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;

int main() { 
  try {
    buffer<int> B{ range{16} };

    // ERROR: Create sub-buffer larger than size of parent buffer
    // An exception is thrown from within the buffer constructor
    buffer<int> B2(B, id{8}, range{16});

  } catch (sycl::exception &e) {
    // Do something to output or handle the exception 
    std::cout << "Caught sync SYCL exception: " << e.what() << "\n";
    return 1;
  } catch (std::exception &e) {
    std::cout << "Caught std exception: " << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cout << "Caught unknown exception\n";
    return 3;
  }

  return 0;
}

