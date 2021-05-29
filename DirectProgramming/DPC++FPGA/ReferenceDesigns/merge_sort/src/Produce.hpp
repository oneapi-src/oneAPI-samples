#ifndef __PRODUCE_HPP__
#define __PRODUCE_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

//
// Produces 'k_width' elements of data per cycle into the merge unit from
// device memory
//
template <typename Id, typename ValueT, typename IndexT, typename OutPipe,
          unsigned char k_width>
event Produce(queue& q, ValueT* in_ptr, IndexT total_count,
              IndexT in_block_count, IndexT start_offset,
              std::vector<event>& depend_events) {
  // producer always produces half of the total count
  const IndexT half_total_count = total_count / 2;

  // number of input blocks to produce
  const IndexT num_blocks = half_total_count / in_block_count;

  // a producer produces a single block and then steps over an entire block
  const IndexT in_block_step = in_block_count * 2;

  return q.submit([&](handler& h) {
    h.depends_on(depend_events);

    h.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<ValueT> in(in_ptr);
      IndexT block_idx = 0;     // the index of the current block
      IndexT block_offset = 0;  // the offset to the start of the current block
      IndexT inter_block_offset = 0;  // the offset within the current block

      while (block_idx != num_blocks) {
        // get the input data from either the input pipe, or device memory
        sycl::vec<ValueT, k_width> pipe_data;
        #pragma unroll
        for (unsigned char j = 0; j < k_width; j++) {
          pipe_data[j] =
              in[start_offset + block_offset + inter_block_offset + j];
        }

        // write to the output pipe
        OutPipe::write(pipe_data);

        // move to the next input
        if (inter_block_offset == in_block_count - k_width) {
          block_idx++;
          block_offset += in_block_step;
          inter_block_offset = 0;
        } else {
          inter_block_offset += k_width;
        }
      }
    });
  });
}

#endif /* __PRODUCE_HPP__ */