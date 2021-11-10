//
// This file contains all of the FPGA device code for the single-kernel design
//

#ifndef __SINGLE_KERNEL_HPP__
#define __SINGLE_KERNEL_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// Forward declare the kernel names to reduce name mangling
class K;

// submit the kernel for the single-kernel design
template<typename T>
event SubmitSingleWorker(queue &q, T *in_ptr, T *out_ptr, size_t count) {
  return q.single_task<K>([=]() [[intel::kernel_args_restrict]] {
    // using a host_ptr class tells the compiler that this pointer lives in
    // the hosts address space
    host_ptr<T> in(in_ptr);
    host_ptr<T> out(out_ptr);

    for (size_t i = 0; i < count; i++) {
      // do a simple copy - more complex computation can go here
      T data = *(in + i);
      *(out + i) = data;
    }
  });
}

#endif /* __SINGLE_KERNEL_HPP__ */
