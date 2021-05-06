#ifndef __PARTITION_HPP__
#define __PARTITION_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

//
// Partition data between pipe A and pipe B
//
template<typename Id, typename ValueT, typename IndexT, typename InPipe,
         typename APipe, typename BPipe>
event Partition(queue& q, IndexT total_count) {
  return q.submit([&](handler& h) {
    h.single_task<Id>([=] {
      bool write_a = true;
      for (IndexT i = 0; i < total_count; i++) {
        // read from the input pipe
        auto d = InPipe::read();

        // write to either APipe or BPipe
        if (write_a) {
          APipe::write(d);
        } else {
          BPipe::write(d);
        }

        // move to the other pipe
        write_a = !write_a;
      }
    });
  });
}

#endif /* __PARTITION_HPP__ */