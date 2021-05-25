#ifndef __CONSUME_HPP__
#define __CONSUME_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

//
// Streams in data from a SYCL pipe and either writes it to memory
// (to_pipe == false) or writes it to a pipe (to_pipe == true)
//
template<typename Id, typename ValueT, typename IndexT,
         typename InPipe, typename OutPipe>
event Consume(queue& q, ValueT *out_ptr, IndexT total_count, IndexT offset,
              bool to_pipe) {
  return q.submit([&](handler& h) {
    h.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<ValueT> out(out_ptr);
      
      for (IndexT i = 0; i < total_count; i++) {
        // read in data from the input pipe
        auto data = InPipe::read();

        // write to either the output pipe or to device memory, depending
        // on a runtime argument
        if (to_pipe) {
          OutPipe::write(data);
        } else {
          out[offset + i] = data;
        }
      }
    });
  });
}

#endif /* __CONSUME_HPP__ */