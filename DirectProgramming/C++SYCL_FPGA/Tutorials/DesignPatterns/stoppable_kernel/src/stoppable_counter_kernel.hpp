#ifndef __STOPPABLE_COUNTER_KERNEL_HPP__
#define __STOPPABLE_COUNTER_KERNEL_HPP__

#include <iostream>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>
#include <sycl/sycl.hpp>

namespace intel_exp = sycl::ext::intel::experimental;
namespace oneapi_exp = sycl::ext::oneapi::experimental;

// Forward declare the pipe names in the global scope. This is an FPGA best
// practice that reduces name mangling in the optimization reports.
class StopPipeID;
class OutputPipeID;

namespace stoppable_counter {

// use sideband signals to signal when a kernel has been reset
using OutputBeat =
    intel_exp::StreamingBeat<int,     // payload of this Avalon streaming
                                      // interface's data signal
                             true,    // enable startofpacket and endofpacket
                                      // signals
                             false>;  // disable the empty signal>;

// This pipe terminates in the CSR. This way a kernel can respond to CSR
// updates while it executes.
using CsrPipeProperties = decltype(oneapi_exp::properties(
    intel_exp::protocol<
        // Write-only, so no no need for protocol_name::avalon_mm_uses_ready
        intel_exp::protocol_name::avalon_mm_uses_ready>));
using StopPipe = intel_exp::pipe<StopPipeID, bool, 0, CsrPipeProperties>;

using StreamingPipeProperties =
    decltype(oneapi_exp::properties(intel_exp::bits_per_symbol<32>));
// This pipe terminates in a streaming interface.
using OutputPipe =
    intel_exp::pipe<OutputPipeID, OutputBeat, 0, StreamingPipeProperties>;

struct StoppableCounter {
  // the kernel argument will be re-read each time the kernel is started.
  int counter_start;

  void operator()() const {
    int counter = counter_start;
    bool keep_going = true;

    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
    while (keep_going) {
      // Use non-blocking operations to ensure that the kernel can check all its
      // pipe interfaces every clock cycle, even if one or more data interfaces
      // are stalling (asserting valid = 0) or back-pressuring (asserting ready
      // = 0).
      bool did_write_counter = false;
      bool start_of_packet = (counter == counter_start);
      bool end_of_packet = false;
      OutputBeat beat(counter, start_of_packet, end_of_packet);
      OutputPipe::write(beat, did_write_counter);

      // Only adjust the state of the kernel if the pipe write succeeded.
      // This is logically equivalent to blocking.
      if (did_write_counter) {
        counter++;
      }

      // Use non-blocking operations to ensure that the kernel can check all its
      // pipe interfaces every clock cycle.
      bool did_read_keep_going = false;
      bool stop_result = StopPipe::read(did_read_keep_going);
      if (did_read_keep_going) {
        keep_going = !stop_result;
      }
    }
  }
};
}  // namespace stoppable_counter
#endif