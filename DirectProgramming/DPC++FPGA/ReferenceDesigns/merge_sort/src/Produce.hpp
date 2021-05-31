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
              IndexT in_sublist_count, IndexT start_offset,
              std::vector<event>& depend_events) {
  // producer always produces half of the total count
  const IndexT half_total_count = total_count / 2;

  // number of input sublists to produce
  const IndexT num_sublists = half_total_count / in_sublist_count;

  // this kernel produces a single sublist and then steps over an entire sublist
  const IndexT in_block_step = in_sublist_count * 2;

  return q.submit([&](handler& h) {
    h.depends_on(depend_events);

    h.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
      device_ptr<ValueT> in(in_ptr);
      IndexT sublist_idx = 0; // index of the current sublist
      IndexT sublist_offset = 0; // offset to the start of the current sublist
      IndexT inter_sublist_offset = 0; // offset within the current sublist

      while (sublist_idx != num_sublists) {
        // get the input data from either the input pipe, or device memory
        sycl::vec<ValueT, k_width> pipe_data;
        #pragma unroll
        for (unsigned char j = 0; j < k_width; j++) {
          pipe_data[j] =
              in[start_offset + sublist_offset + inter_sublist_offset + j];
        }

        // write to the output pipe
        OutPipe::write(pipe_data);

        // move to the next sublist
        if (inter_sublist_offset == in_sublist_count - k_width) {
          sublist_idx++;
          sublist_offset += in_block_step;
          inter_sublist_offset = 0;
        } else {
          inter_sublist_offset += k_width;
        }
      }
    });
  });
}

#endif /* __PRODUCE_HPP__ */