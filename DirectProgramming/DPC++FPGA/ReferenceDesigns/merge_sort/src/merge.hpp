#ifndef __MERGE_HPP__
#define __MERGE_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "sorting_networks.hpp"
#include "impu_math.hpp"

using namespace sycl;

//
// Streams in two sorted list of size 'in_count`, 'k_width' elements at a time,
// from both InPipeA and InPipeB and merges them into a single sorted list of
// size 'in_count*2' to OutPipe. This merges two sorted lists of size in_count
// at a rate of 'k_width' elements per cycle.
//
template <typename Id, typename ValueT, typename IndexT, typename InPipeA,
          typename InPipeB, typename OutPipe, unsigned char k_width,
          class CompareFunc>
event Merge(queue& q, IndexT total_count, IndexT in_count,
            CompareFunc compare) {
  // sanity check on k_width
  static_assert(k_width >= 1);
  static_assert(impu::math::IsPow2(k_width));

  // merging two lists of size 'in_count' into a single output list of
  // double the size
  const IndexT out_count = in_count * 2;

  return q.single_task<Id>([=] {
    // the two input and feedback buffers
    sycl::vec<ValueT, k_width> a, b, network_feedback;

    bool drain_a = false;
    bool drain_b = false;
    bool a_valid = false;
    bool b_valid = false;

    // track the number of elements we have read from each input pipe
    // for each sublist (counts up to 'in_count')
    IndexT read_from_a = 0;
    IndexT read_from_b = 0;

    // create a small 2 element shift register to track whether we have
    // read the last inputs from the input pipes
    bool read_from_a_is_last = false; // (0 == in_count)
    bool read_from_b_is_last = false; // (0 == in_count)
    bool next_read_from_a_is_last = (k_width == in_count);
    bool next_read_from_b_is_last = (k_width == in_count);

    // track the number of elements we have written to the output pipe
    // for each sublist (counts up to 'out_count')
    IndexT written_out_inner = 0;

    // track the number of elements we have written to the output pipe
    // in total (counts up to 'total_count')
    IndexT written_out = 0;

    // this flag indicates that the chosen buffer (from Pipe A or B) is the
    // first buffer from either sublist. This indicates that no output will
    // be produced and instead we will just populate the feedback buffer
    bool first_in_buffer = true;

    // the main processing loop
    [[intel::initiation_interval(1)]]
    while (written_out != total_count) {
      // read 'k_width' elements from Pipe A
      if (!a_valid && !drain_b) {
        a = InPipeA::read();
        a_valid = true;
        read_from_a_is_last = next_read_from_a_is_last;
        next_read_from_a_is_last = (read_from_a == in_count-2*k_width);
        read_from_a += k_width;
      }

      // read 'k_width' elements from Pipe B
      if (!b_valid && !drain_a) {
        b = InPipeB::read();
        b_valid = true;
        read_from_b_is_last = next_read_from_b_is_last;
        next_read_from_b_is_last = (read_from_b == in_count-2*k_width);
        read_from_b += k_width;
      }

      // determine which of the two inputs to feed into the merge sort network
      bool choose_a = ((compare(a[0], b[0]) || drain_a) && !drain_b);
      auto chosen_data_in = choose_a ? a : b;

      // create input for merge sort network sorter network
      sycl::vec<ValueT, k_width * 2> merge_sort_network_data;
      #pragma unroll
      for (unsigned char i = 0; i < k_width; i++) {
        // populate the k_width*2 sized input for the merge sort network
        // from the chosen input data and the feedback data
        merge_sort_network_data[2 * i] = chosen_data_in[i];
        merge_sort_network_data[2 * i + 1] = network_feedback[i];
      }

      // merge sort network, which sorts 'merge_sort_network_data' in-place
      MergeSortNetwork<ValueT, k_width>(merge_sort_network_data, compare);

      if (first_in_buffer) {
        // the first buffer read for a sublist doesn't create any output,
        // it just creates feedback
        #pragma unroll
        for (unsigned char i = 0; i < k_width; i++) {
          network_feedback[i] = chosen_data_in[i];
        }
        drain_a = drain_a | (read_from_b_is_last && !choose_a);
        drain_b = drain_b | (read_from_a_is_last && choose_a);
        a_valid = !choose_a;
        b_valid = choose_a;
        first_in_buffer = false;
      } else {
        sycl::vec<ValueT, k_width> out_data;
        if (written_out_inner == out_count - k_width) {
          // on the last iteration for a set of sublists, the feedback
          // is the only data left that is valid, so it goes to the output
          out_data = network_feedback;
        } else {
          // grab the output and feedback data from the merge sort network
          #pragma unroll
          for (unsigned char i = 0; i < k_width; i++) {
            out_data[i] = merge_sort_network_data[i];
            network_feedback[i] = merge_sort_network_data[k_width + i];
          }
        }

        // write the output data to the output pipe
        OutPipe::write(out_data);
        written_out += k_width;

        // check if switching to a new set of 'in_count' sorted sublists
        if (written_out_inner == out_count - k_width) {
          // switching, so reset all internal counters and flags
          drain_a = false;
          drain_b = false;
          a_valid = false;
          b_valid = false;
          read_from_a = 0;
          read_from_b = 0;
          read_from_a_is_last = false; // (0 == in_count)
          read_from_b_is_last = false; // (0 == in_count)
          next_read_from_a_is_last = (k_width == in_count);
          next_read_from_b_is_last = (k_width == in_count);
          written_out_inner = 0;
          first_in_buffer = true;
        } else {
          // not switching, so update counters and flags
          written_out_inner += k_width;
          drain_a = drain_a | (read_from_b_is_last && !choose_a);
          drain_b = drain_b | (read_from_a_is_last && choose_a);
          a_valid = !choose_a;
          b_valid = choose_a;
        }
      }
    }
  });
}

#endif /* __MERGE_HPP__ */
