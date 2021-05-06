#ifndef __PRODUCE_HPP__
#define __PRODUCE_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

//
// Produces data into the merge unit either from an input pipe (from_pipe==true)
// or from memory (from_pipe==false).
//
template<typename Id, typename ValueT, typename IndexT, typename InPipe,
         typename OutPipe>
event Produce(queue& q, ValueT *in_ptr, IndexT total_count,
              IndexT in_block_count, bool from_pipe,
              std::vector<event>& depend_events) {
  // producer always produces half of the total count
  const IndexT half_total_count = total_count / 2;

  // number of input blocks to produce
  const IndexT num_blocks = half_total_count / in_block_count;

  // a producer produces a single block and then steps over an entire block
  // to produce the next block
  const IndexT in_block_step = in_block_count*2;

  return q.submit([&](handler& h) {
    h.depends_on(depend_events);

    h.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<ValueT> in(in_ptr);

      IndexT block_idx = 0;  // the index of the current block
      IndexT block_offset = 0;  // the offset to the start of the current block
      IndexT inter_block_offset = 0;  // the offset within the current block

      while (block_idx != num_blocks) {
        // get the input data from either the input pipe, or device memory
        ValueT data;
        if (from_pipe) {
          data = InPipe::read();
        } else {
          data = *(in + block_offset + inter_block_offset);
        }

        // write to the output pipe
        OutPipe::write(data);

        // move to the next input
        if (inter_block_offset == in_block_count-1) {
          // move to the next block
          block_idx++;
          block_offset += in_block_step;
          inter_block_offset = 0;
        } else {
          // move within the current block
          inter_block_offset++;
        }
      }
    });
  });
}

//
// Same as the produce about, but with no input pipe
//
template<typename Id, typename ValueT, typename IndexT, typename OutPipe>
event Produce(queue& q, ValueT *in_ptr, IndexT total_count,
              IndexT in_block_count, std::vector<event>& depend_events) {
  // producer always produces half of the total count
  const IndexT half_total_count = total_count / 2;

  // number of input blocks to produce
  const IndexT num_blocks = half_total_count / in_block_count;

  // a producer produces a single block and then steps over an entire block
  const IndexT in_block_step = in_block_count*2;

  return q.submit([&](handler& h) {
    h.depends_on(depend_events);

    h.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<ValueT> in(in_ptr);
      IndexT block_idx = 0;  // the index of the current block
      IndexT block_offset = 0;  // the offset to the start of the current block
      IndexT inter_block_offset = 0;  // the offset within the current block

      while (block_idx != num_blocks) {
        // get the input data from either the input pipe, or device memory
        ValueT data = *(in + block_offset + inter_block_offset);

        // write to the output pipe
        OutPipe::write(data);

        // move to the next input
        if (inter_block_offset == in_block_count-1) {
          block_idx++;
          block_offset += in_block_step;
          inter_block_offset = 0;
        } else {
          inter_block_offset++;
        }
      }
    });
  });
}

#endif /* __PRODUCE_HPP__ */