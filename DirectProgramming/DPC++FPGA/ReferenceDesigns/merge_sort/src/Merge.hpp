#ifndef __MERGE_HPP__
#define __MERGE_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;

//
// Streams in a sorted list of size 'in_count` from both InPipeA and InPipeB
// and merges them into a single sorted list of size 'in_count*2' to OutPipe.
// Does this for total_count/(in_count*2) iterations.
//
template<typename Id, typename ValueT, typename IndexT, typename InPipeA,
         typename InPipeB, typename OutPipe, class CompareFunc>
event Merge(queue& q, IndexT total_count, IndexT in_count,
            CompareFunc compare) {
  // the output size is 2x the input size, since we are merging 2 sorted lists
  // of size 'count' into a single sorted list
  // NOTE: this is NOT computed on the FPGA, it is an argument to the kernel
  const IndexT out_size = in_count * 2;

  return q.submit([&](handler& h) {
    h.single_task<Id>([=] {
      // the values read from pipe A and B, respectively
      ValueT a, b;

      // signals that it is time to drain input pipe A and B, respectively
      bool drain_a = false;
      bool drain_b = false;

      // signals if the 'a' and 'b' inputs are valid, respectively
      bool a_valid = false;
      bool b_valid = false;

      // track the number of elements read from each input pipe
      IndexT read_from_a = 0;
      IndexT read_from_b = 0;

      // track the number of elements written to the output pipe for one
      // output sub-array
      IndexT written_out_inner = 0;

      // track the total number of elements written to the output pipe
      IndexT written_out = 0;

      // signals that the current read is the last read from each input pipe
      bool read_from_a_is_last = false;
      bool read_from_b_is_last = false;

      // signals that the NEXT read will be the last read from each input pipe
      bool next_read_from_a_is_last = (in_count == 1);
      bool next_read_from_b_is_last = (in_count == 1);

      // repeat until we have seen all of the elements to sort
      while (written_out != total_count) {
        // read from InPipeA if 'a' is not valid and we aren't in the process
        // of draining InPipeB. Vice versa for InPipeB.
        if (!a_valid && !drain_b) {
          // read from pipe A and mark the input as valid
          a = InPipeA::read();
          a_valid = true;

          // 2-element shift register to track whether the element read from
          // pipe A was the last element to read from the input pipe before
          // draining pipe B. This shift register removes an addition and
          // comparison from the critical path
          read_from_a_is_last = next_read_from_a_is_last;
          next_read_from_a_is_last = (read_from_a == in_count-2);
          read_from_a++;
        }
        if (!b_valid && !drain_a) {
          // read from pipe B and mark the input as valid
          b = InPipeB::read();
          b_valid = true;

          // same technique as the if-case
          read_from_b_is_last = next_read_from_b_is_last;
          next_read_from_b_is_last = (read_from_b == in_count-2);
          read_from_b++;
        }

        // determine which element we want to output
        bool choose_a = ((compare(a, b) || drain_a) && !drain_b);
        ValueT out_data = choose_a ? a : b;

        // write the output
        OutPipe::write(out_data);
        written_out++;

        // check if we are switching to a new set of 'in_count' inputs
        if (written_out_inner == out_size-1) {
          // switching, so reset all internal counters and flags
          drain_a = false;
          drain_b = false;
          a_valid = false;
          b_valid = false;
          read_from_a = 0;
          read_from_b = 0;
          read_from_a_is_last = false;
          read_from_b_is_last = false;
          next_read_from_a_is_last = (in_count == 1);
          next_read_from_b_is_last = (in_count == 1);
          written_out_inner = 0;
        } else {
          // increment the internal counter
          written_out_inner++;

          // determine whether we are draining input pipe A or B, respectively
          drain_a = drain_a | (read_from_b_is_last && !choose_a);
          drain_b = drain_b | (read_from_a_is_last && choose_a);

          // if we chose 'a', it is no longer valid, otherwise 'b' is invalid
          a_valid = !choose_a;
          b_valid = choose_a;
        }
      }
    });
  });
}

#endif /* __MERGE_HPP__ */