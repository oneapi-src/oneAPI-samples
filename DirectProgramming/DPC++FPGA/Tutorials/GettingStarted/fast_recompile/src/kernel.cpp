//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "kernel.hpp"


// This file contains 'almost' exclusively device code. The single-source SYCL
// code has been refactored between host.cpp and kernel.cpp to separate host and
// device code to the extent that the language permits.
//
// Note that ANY change in either this file or in kernel.hpp will be detected
// by the build system as a difference in the dependencies of device.o,
// triggering a full recompilation of the device code. 
//
// This is true even of trivial changes, e.g. tweaking the function definition 
// or the names of variables like 'q' or 'h', EVEN THOUGH these are not truly 
// "device code".


// Forward declare the kernel names in the global scope. This FPGA best practice
// reduces compiler name mangling in the optimization reports.
class VectorAdd;

void RunKernel(queue& q, buffer<float,1>& buf_a, buffer<float,1>& buf_b,
               buffer<float,1>& buf_r, size_t size){
    // submit the kernel
    q.submit([&](handler &h) {
      // Data accessors
      accessor a(buf_a, h, read_only);
      accessor b(buf_b, h, read_only);
      accessor r(buf_r, h, write_only, noinit);

      // Kernel executes with pipeline parallelism on the FPGA.
      // Use kernel_args_restrict to specify that a, b, and r do not alias.
      h.single_task<VectorAdd>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < size; ++i) {
          r[i] = a[i] + b[i];
        }
      });
    });
}
