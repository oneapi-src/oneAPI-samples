// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include<CL/sycl.hpp>
#include<iostream>
using namespace sycl;

int main() {
  queue Q;
  Q.submit([&](handler &h){
    stream out(1024, 256, h);
    h.parallel_for(range{8}, [=](id<1> idx){
      out << "Testing my sycl stream (this is work-item ID:" << idx << ")\n";
    });
  });

  // Wait on the queue so that the host program doesn't complete before the device
  // code stream out is executed.  This ensures that the example actually displays
  // the output text.
  Q.wait();

  return 0;
}

