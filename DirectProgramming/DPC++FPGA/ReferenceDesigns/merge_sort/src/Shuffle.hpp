#ifndef __SHUFFLE_HPP__
#define __SHUFFLE_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "UnrolledLoop.hpp"

using namespace sycl;

//
// Shuffle data between pipe A and pipe B across the merge units
//
template<typename Id, typename ValueT, typename IndexT, typename InPipe,
         typename APipes, typename BPipes, size_t units>
event Shuffle(queue& q, IndexT total_size) {
  // the number of elements per merge unit
  // NOTE: this is NOT computed on the FPGA, it is an argument to the kernel
  const IndexT count_per_unit = total_size / units;

  return q.submit([&](handler& h) {
    h.single_task<Id>([=] {
      bool write_a = 0;
      unsigned char current_unit = 0;
      IndexT unit_counter = 0;

      for (IndexT i = 0; i < total_size; i++) {
        // read the data from the input pipe
        auto data = InPipe::read();

        // write it to the current output pipe
        if (write_a) {
          // writing to pipe A of the current merge unit
          UnrolledLoop<units>([&](auto u) {
            if (u == current_unit) APipes::template pipe<u>::write(data);
          });
        } else {
          // writing to pipe B of the current merge unit
          UnrolledLoop<units>([&](auto u) {
            if (u == current_unit) BPipes::template pipe<u>::write(data);
          });
        }

        // shuffle between A and B pipe on every iteration
        write_a = !write_a;

        // shuffle to each of the merge units
        if (unit_counter == count_per_unit-1) {
          // time to switch to the next merge unit
          unit_counter = 0;
          current_unit++;
        } else {
          unit_counter++;
        }
      }
    });
  });
}

#endif /* __SHUFFLE_HPP__ */