// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <mutex>
using namespace sycl;

int main() {
  queue Q;
  int my_ints[42];

  // create a buffer of 42 ints
  buffer<int> b{range(42)};

  // create a buffer of 42 ints, initialize with a host pointer,
  // and add the use_host_pointer property
  buffer b1{my_ints, range(42), {property::buffer::use_host_ptr{}}};

  // create a buffer of 42 ints, initialize with a host pointer,
  // and add the use_mutex property
  std::mutex myMutex;
  buffer b2{my_ints, range(42), {property::buffer::use_mutex{myMutex}}};
  // Retrive a pointer to the mutex used by this buffer
  auto mutexPtr = b2.get_property<property::buffer::use_mutex>().get_mutex_ptr();
  // lock the mutex until we exit scope
  std::lock_guard<std::mutex> guard{*mutexPtr};

  // create a context-bound buffer of 42 ints, initialized from a host pointer
  buffer b3{my_ints, range(42), {property::buffer::context_bound{Q.get_context()}}};

  return 0;
}
