#ifndef __CONSUME_HPP__
#define __CONSUME_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

//
// Streams in 'k_width' elements of data per cycle from a SYCL pipe and either
// writes it to memory (to_pipe==false) or writes it to a pipe (to_pipe==true)
//
template <typename Id, typename ValueT, typename IndexT, typename InPipe,
          typename OutPipe, unsigned char k_width>
event Consume(queue& q, ValueT* out_ptr, IndexT total_count, IndexT offset,
              bool to_pipe) {
  // the number of loop iterations required to consume all of the data
  const IndexT iterations = total_count / k_width;

  return q.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
    // Pointer to the output data.
    // Creating a device_ptr tells the compiler that this pointer is in
    // device memory, not host memory, and avoids creating extra connections
    // to host memory
    // This is only done in the case where we target a BSP as device 
    // pointers are not supported when targeting an FPGA family/part
#if defined(IS_BSP)
    device_ptr<ValueT> out(out_ptr);
#else
    ValueT* out(out_ptr);
#endif

    for (IndexT i = 0; i < iterations; i++) {
      // get the data from the pipe
      auto data = InPipe::read();

      // write to either the output pipe, or to device memory
      if (to_pipe) {
        OutPipe::write(data);
      } else {
        // write the 'k_width' elements to device memory
        #pragma unroll
        for (unsigned char j = 0; j < k_width; j++) {
          out[offset + i * k_width + j] = data[j];
        }
      }
    }
  });
}

#endif /* __CONSUME_HPP__ */
