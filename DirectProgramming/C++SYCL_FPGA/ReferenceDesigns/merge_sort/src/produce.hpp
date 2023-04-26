#ifndef __PRODUCE_HPP__
#define __PRODUCE_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

//
// Produces 'k_width' elements of data per cycle into the merge unit from
// device memory
//
template<typename Id, typename ValueT, typename IndexT, typename OutPipe,
         unsigned char k_width>
event Produce(queue& q, ValueT *in_ptr, IndexT count, IndexT in_block_count,
              IndexT start_offset, std::vector<event>& depend_events) {
  // the number of loop iterations required to produce all of the data
  const IndexT iterations = count / k_width;

  return q.submit([&](handler& h) {
    h.depends_on(depend_events);
    h.single_task<Id>([=]() [[intel::kernel_args_restrict]] {
      // Pointer to the input data.
      // Creating a device_ptr tells the compiler that this pointer is in
      // device memory, not host memory, and avoids creating extra connections
      // to host memory
      // This is only done in the case where we target a BSP as device 
      // pointers are not supported when targeting an FPGA family/part
#if defined(IS_BSP)
      device_ptr<ValueT> in(in_ptr);
#else
      ValueT* in(in_ptr);
#endif

      for (IndexT i = 0; i < iterations; i++) {
        // read 'k_width' elements from device memory
        sycl::vec<ValueT, k_width> pipe_data;
        #pragma unroll
        for (unsigned char j = 0; j < k_width; j++) {
          pipe_data[j] = in[start_offset + i*k_width + j];
        }

        // write to the output pipe
        OutPipe::write(pipe_data);
      }
    });
  });
}

#endif /* __PRODUCE_HPP__ */
