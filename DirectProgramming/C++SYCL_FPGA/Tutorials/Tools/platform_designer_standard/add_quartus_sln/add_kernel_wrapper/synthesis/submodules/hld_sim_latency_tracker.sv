// (c) 1992-2024 Intel Corporation.                            
// Intel, the Intel logo, Intel, MegaCore, NIOS II, Quartus and TalkBack words    
// and logos are trademarks of Intel Corporation or its subsidiaries in the U.S.  
// and/or other countries. Other marks and brands may be claimed as the property  
// of others. See Trademarks on intel.com for full list of Intel trademarks or    
// the Trademarks & Brands Names Database (if Intel) or See www.Intel.com/legal (if Altera) 
// Your use of Intel Corporation's design tools, logic functions and other        
// software and tools, and its AMPP partner logic functions, and any output       
// files any of the foregoing (including device programming or simulation         
// files), and any associated documentation or information are expressly subject  
// to the terms and conditions of the Altera Program License Subscription         
// Agreement, Intel MegaCore Function License Agreement, or other applicable      
// license agreement, including, without limitation, that your use is for the     
// sole purpose of programming logic devices manufactured by Intel and sold by    
// Intel or its authorized distributors.  Please refer to the applicable          
// agreement for further details.                                                 


// This IP is non-synthesizable and use for oneAPI, HLS, OpenCL cosim to measure latency
// for component, loop, block and IP/instructions user wants to measure latency.
// The IP assumes incoming threads are executed in order.
// Example usage:
// +-------------------+-------------------------+------------------------+
// | Hierarchy         | i_start                 | i_end                  |
// +-------------------+-------------------------+------------------------+
// | Component         | read_implicit_streams   | is_done                |
// | Block/simple loop | i_valid_predr &         | i_valid_succr &        |
// |                   |   !i_stall_predr        |   !i_stall_succr       |
// | Loop nest         | i_valid_header_predr &  | i_valid_latch_succr &  |
// |                   |   !i_stall_header_predr |   !i_stall_latch_succr |
// | LSU IP            | i_valid & !i_stall      | o_valid & !o_stall     |
// +-------------------+-------------------------+------------------------+
//
// Parameters:
//  NAME : unique string for identification. Could be component name, block name, or
//         instruction name
//  IS_COMPONENT : 1 means this is a component, default is 0 and it means no.
//                 This parameter has a default because the only
//                 hls_sim_component_dpi_controller IP needs to overwrite the value.
//                 The compiler should not be setting this value.
//  CRA_CONTROL:   A value of 1 indicates that this the IP is tracking the
//                 latency of a component with a CRA used for control, rather
//                 than a streaming control interface.
//  ADDITIONAL_START_LATENCY: This parameter indicates the number of cycles
//                 (if any) that should be subtracted from the start_time
//                 timestamp.  Used when CRA_CONTROL=1 to compensate for the
//                 time between when the start signal is written to the CRA
//                 and the start signal reaches the latency tracker IP.
//

`default_nettype none
module hld_sim_latency_tracker #(
  parameter string NAME,
  parameter int IS_COMPONENT = 0,
  parameter int CRA_CONTROL = 0,
  parameter int ADDITIONAL_START_LATENCY = 0
)
(
  input  wire clock,
  input  wire resetn,

  input  wire i_enable,
  input  wire i_start,
  input  wire i_end
);
//synthesis translate_off
`ifdef COSIM_LIB
  import "DPI-C" context function void __ihc_hls_dbgs(string msg);
  import "DPI-C" context function void __ihc_hls_register_component_invocation_info(string component_name, longint unsigned start_time, longint unsigned end_time, longint unsigned  concurrent_threads, longint unsigned first_in_set);
  import "DPI-C" context function void __ihc_register_inst_invocation_info(string component_name, longint unsigned start_time, longint unsigned end_time, longint unsigned  concurrent_threads, longint unsigned first_in_set, int unsigned is_component, int unsigned start_offset);

  // stat tracking for each thread entering into this loop or component
  typedef struct {
    time start_time;
    longint unsigned thread_index;  // index of current thread
    longint unsigned first_in_set;
  } invocation_stats;
  invocation_stats queue[$] = {};  // TODO: don't use a queue if MAX_INTERLEAVE > 1

  string message;
  longint unsigned first_in_set;
  longint unsigned num_threads;  // number of threads in this module
  invocation_stats istat_in, istat_out;

  wire the_start_sig;

  if (CRA_CONTROL) begin
    // If this component is being controlled by a CRA, we will see a 2-cycle
    // start signal.  Ensure we only register a single invocation.
    reg i_start_reg;
    always @(posedge clock) begin
      i_start_reg <= i_start;
    end
    assign the_start_sig = i_start & i_start_reg;
  end else begin
    assign the_start_sig = i_start;
  end

  initial begin
    num_threads  = 0;
    first_in_set = 1;
    forever begin
      @(posedge clock);  // make everything else sync to clock
      if (!resetn) begin
        num_threads  = 0;
        first_in_set = 1;
      end
      else begin
        if (!i_enable) begin
          first_in_set = 1;
        end else begin
          // save start info
          if (the_start_sig) begin
            istat_in.start_time         = $time;
            istat_in.thread_index       = num_threads;
            istat_in.first_in_set       = first_in_set;
            queue.push_back(istat_in);
            $sformat(message, "[%7t][msim][sim_latency_tracker] push thread=%d", $time, istat_in.thread_index);
            __ihc_hls_dbgs(message);
            num_threads  = num_threads + 1;
            if (!CRA_CONTROL) begin
              first_in_set = 0;
            end
          end

          // register info for this invocation
          if (i_end) begin
            istat_out = queue.pop_front();
            $sformat(message, "[%7t][msim][sim_latency_tracker] pop thread=%d", $time, istat_out.thread_index);
            __ihc_hls_dbgs(message);
            if (IS_COMPONENT) begin
              // For HLS to create verification statistics viewer
              __ihc_hls_register_component_invocation_info(NAME, istat_out.start_time, $time, istat_out.thread_index, istat_out.first_in_set);
            end
            __ihc_register_inst_invocation_info(NAME, istat_out.start_time, $time, istat_out.thread_index, istat_out.first_in_set, IS_COMPONENT, ADDITIONAL_START_LATENCY);
            num_threads = num_threads - 1;
          end
        end
      end
    end
  end
`endif  // COSIM_LIB
//synthesis translate_on

endmodule

`default_nettype wire
