#ifndef __CONSUME_HPP__
#define __CONSUME_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

//
// Streams in 'k_width' elements of data per cycle from a SYCL pipe and either
// writes it to memory (to_pipe==false) or writes it to a pipe (to_pipe==true)
//
template <typename Id, typename ValueT, typename IndexT, typename InPipe,
          typename OutPipe, unsigned char k_width>
event Consume(queue& q, ValueT* out_ptr, IndexT total_count, IndexT offset,
              bool to_pipe) {
  const IndexT iterations = total_count / k_width;

  return q.submit([&](handler& h) {
    h.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<ValueT> out(out_ptr);
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
  });
}

#endif /* __CONSUME_HPP__ */