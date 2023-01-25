#ifndef __BUFFER_KERNEL_HPP__
#define __BUFFER_KERNEL_HPP__
#pragma once

#include <vector>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;
using namespace std::chrono;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class BufferWorker;

// kernel using buffers to transfer data
template <typename T>
double SubmitBufferKernel(queue& q, std::vector<T>& in, std::vector<T>& out,
                          const unsigned int size) {
  // start timer
  auto start = high_resolution_clock::now();

  {
    // set up the input/output buffers
    buffer in_buf(in);
    buffer out_buf(out);

    // launch the computation kernel
    auto kernel_event = q.submit([&](handler& h) {

#if defined (IS_BSP)
      accessor in_a(in_buf, h, read_only);
      accessor out_a(out_buf, h, write_only, no_init);
#else
      // When targeting an FPGA family/part, the compiler does not know
      // if the two kernels accesses the same memory location
      // With this property, we tell the compiler that these buffers
      // are in a location "1" whereas the pointers from ExplicitKernel
      // are in the default location "0"
      sycl::ext::oneapi::accessor_property_list location_of_buffer{
          ext::intel::buffer_location<1>};
      accessor in_a(in_buf, h, read_only, location_of_buffer);

      sycl::ext::oneapi::accessor_property_list location_of_buffer_no_init{
          no_init, ext::intel::buffer_location<1>};
      accessor out_a(out_buf, h, write_only, location_of_buffer_no_init);
#endif


      h.single_task<BufferWorker>([=]() [[intel::kernel_args_restrict]] {
        for (size_t i = 0; i < size; i++) {
          out_a[i] = in_a[i] * i;
        }
      });
    });
  }

  // We use the scope above to synchronize the FPGA kernels.
  // Exiting the scope will cause the buffer destructors to be called
  // which will wait until the kernels finish and copy the data back to the
  // host (if the buffer was written to).
  // Therefore, at this point in the code, we know the kernels have finished
  // and the data has been transferred back to the host.

  // stop the timer
  auto end = high_resolution_clock::now();
  duration<double, std::milli> diff = end - start;

  return diff.count();
}

#endif /* __BUFFER_KERNEL_HPP__ */
