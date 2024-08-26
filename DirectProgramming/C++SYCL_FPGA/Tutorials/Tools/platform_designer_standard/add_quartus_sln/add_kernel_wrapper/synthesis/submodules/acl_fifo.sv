//// (c) 1992-2024 Intel Corporation.                            
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


//===----------------------------------------------------------------------===//
//
// Parameterized FIFO with input and output registers and ACL pipeline
// protocol ports.
//
//===----------------------------------------------------------------------===//

`default_nettype none

module acl_fifo (
	clock,
	resetn,
	data_in,
	data_out,
	valid_in,
	valid_out,
	stall_in,
	stall_out,
	usedw,
	empty,
	full,
	almost_full,
  ecc_err_status);

	function integer my_local_log;
	input [31:0] value;
		for (my_local_log=0; value>0; my_local_log=my_local_log+1)
			value = value>>1;
	endfunction		
	
	parameter DATA_WIDTH = 32;
	parameter DEPTH = 256;
	parameter NUM_BITS_USED_WORDS = DEPTH == 1 ? 1 : my_local_log(DEPTH-1);
	parameter ALMOST_FULL_VALUE = 0;
	parameter LPM_HINT = "unused";
    parameter string IMPL = "basic";   // impl: (basic|pow_of_2_full|pow_of_2_full_reg_data_in|pow_of_2_full_reg_output_accepted|pow_of_2_full_reg_data_in_reg_output_accepted)
    parameter bit ASYNC_RESET = 1;          // how do the registers CONSUME reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 0;    // before consumption, do we SYNCHRONIZE the reset: 1 means use a synchronizer (assume reset arrived asynchronously), 0 means passthrough (assume reset was already synchronized)
    parameter enable_ecc = "FALSE";               // Enable error correction coding

	input wire clock, stall_in, valid_in, resetn;
	output logic stall_out, valid_out;
	input wire [DATA_WIDTH-1:0] data_in;
	output logic [DATA_WIDTH-1:0] data_out;
	output logic [NUM_BITS_USED_WORDS-1:0] usedw;
    output logic empty, full, almost_full;
    output logic [1:0] ecc_err_status; // ecc status signals
    
    logic aclrn, sclrn;
    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
        .PULSE_EXTENSION        (0),
        .PIPE_DEPTH             (1),
        .NUM_COPIES             (1)
    )
    acl_reset_handler_inst
    (
        .clk                    (clock),
        .i_resetn               (resetn),
        .o_aclrn                (aclrn),
        .o_resetn_synchronized  (),
        .o_sclrn                (sclrn)
    );
    

  generate
    if ((IMPL == "pow_of_2_full") || 
        (IMPL == "pow_of_2_full_reg_output_accepted") ||
        (IMPL == "pow_of_2_full_reg_data_in") ||
        (IMPL == "pow_of_2_full_reg_data_in_reg_output_accepted"))
    begin
          localparam DEPTH_LOG2 = $clog2(DEPTH);
          localparam DEPTH_SNAPPED_TO_POW_OF_2 = 1 << DEPTH_LOG2;
          localparam FULL_COUNTER_OFFSET = DEPTH_SNAPPED_TO_POW_OF_2 - DEPTH;
          
          localparam ALMOST_FULL_DEPTH_LOG2 = $clog2(DEPTH); // required to be DEPTH, this guarantees that almost_full=1 iff fifo occupancy >= ALMOST_FULL_VALUE
          localparam ALMOST_FULL_DEPTH_SNAPPED_TO_POW_OF_2 = 1 << ALMOST_FULL_DEPTH_LOG2;
          localparam ALMOST_FULL_COUNTER_OFFSET = ALMOST_FULL_DEPTH_SNAPPED_TO_POW_OF_2 - ALMOST_FULL_VALUE;

          logic [DEPTH_LOG2:0]              full_counter;
          logic [ALMOST_FULL_DEPTH_LOG2:0]  almost_full_counter;
          
          logic input_accepted_comb;
          logic input_accepted_for_fifo;
          logic input_accepted_for_counter;
          logic output_accepted_comb;
          logic output_accepted_for_fifo;
          logic output_accepted_for_counter;
          logic [DATA_WIDTH-1:0] data_in_for_fifo;
          logic [1:0] counter_update;
          logic [DEPTH_LOG2:0] full_counter_update;
          logic [ALMOST_FULL_DEPTH_LOG2:0] almost_full_counter_update;

          assign full         = full_counter[DEPTH_LOG2];
          assign almost_full  = almost_full_counter[ALMOST_FULL_DEPTH_LOG2];

          assign input_accepted_comb  = valid_in & ~full;
          assign output_accepted_comb = ~stall_in & ~empty;

          assign input_accepted_for_counter = input_accepted_comb;
          assign output_accepted_for_fifo   = output_accepted_comb;

          assign counter_update = input_accepted_for_counter - output_accepted_for_counter;
          // Sign extend to the correct widths
          // TODO: Remove this and replace with $signed when HSD:14021297674 is fixed
          //       and we've stopped supporting all ACDS versions that have the bug
          if (DEPTH_LOG2-2 > 0) begin
            assign full_counter_update = {{(DEPTH_LOG2-2){counter_update[1]}}, counter_update};
          end else begin
            assign full_counter_update = counter_update;
          end

          if (ALMOST_FULL_DEPTH_LOG2-1 > 0) begin
            assign almost_full_counter_update = {{(ALMOST_FULL_DEPTH_LOG2-1){counter_update[1]}}, counter_update};
          end else begin
            assign almost_full_counter_update = counter_update;
          end

          if ((IMPL == "pow_of_2_full") || (IMPL=="pow_of_2_full_reg_data_in"))
          begin
            assign output_accepted_for_counter = output_accepted_comb;
          end
          else // pow_of_2_full_reg_output_accepted, pow_of_2_full_reg_output_accepted_reg_data_in
          begin
            logic stall_in_reg;
            logic empty_reg;

            always @(posedge clock or negedge aclrn)
            begin
              if (~aclrn)
              begin
                stall_in_reg  <= 1;
                empty_reg     <= 1;
              end
              else
              begin
                stall_in_reg  <= stall_in;
                empty_reg     <= empty;
                if (~sclrn)
                begin
                  stall_in_reg  <= 1;
                  empty_reg     <= 1;
                end
              end
            end 
            
            // registered and retimed version of output_accepted_comb
            assign output_accepted_for_counter = ~stall_in_reg & ~empty_reg;
          end

          if ((IMPL == "pow_of_2_full") || (IMPL == "pow_of_2_full_reg_output_accepted")) 
          begin
            assign input_accepted_for_fifo    = input_accepted_comb;
            assign data_in_for_fifo           = data_in;
          end
          else // pow_of_2_full_reg_data_in, pow_of_2_full_reg_output_accepted_reg_data_in
          begin
            logic input_accepted_reg;
            logic [DATA_WIDTH-1:0] data_in_reg;

            always @(posedge clock or negedge aclrn)
            begin
              if (~aclrn)
              begin
                input_accepted_reg  <= 0;
                data_in_reg         <= 'x;
              end
              else
              begin
                input_accepted_reg  <= input_accepted_comb;
                data_in_reg         <= data_in;
                if (~sclrn)
                begin
                  input_accepted_reg  <= 0;
                  data_in_reg         <= 'x;
                end
              end
            end

            assign input_accepted_for_fifo    = input_accepted_reg;
            assign data_in_for_fifo           = data_in_reg;
          end
          
          always @(posedge clock or negedge aclrn)
          begin
            if (~aclrn)
            begin
              full_counter        <= FULL_COUNTER_OFFSET;
              almost_full_counter <= ALMOST_FULL_COUNTER_OFFSET;
            end
            else
            begin
              full_counter        <= full_counter         + full_counter_update; 
              almost_full_counter <= almost_full_counter  + almost_full_counter_update;
              if (~sclrn)
              begin
                full_counter        <= FULL_COUNTER_OFFSET;
                almost_full_counter <= ALMOST_FULL_COUNTER_OFFSET;
              end
            end
          end 
          
          assign usedw = '0;
          assign stall_out = full;
          hld_fifo #(
            .STYLE                          ("ms"), //acl_mid_speed_fifo
            .WIDTH                          (DATA_WIDTH),
            .DEPTH                          (DEPTH),
            .ALMOST_EMPTY_CUTOFF            (0),
            .ALMOST_FULL_CUTOFF             (0),
            .INITIAL_OCCUPANCY              (0),
            .ASYNC_RESET                    (ASYNC_RESET),
            .SYNCHRONIZE_RESET              (SYNCHRONIZE_RESET),
            .RESET_EVERYTHING               (0),
            .RESET_EXTERNALLY_HELD          (1),
            .STALL_IN_EARLINESS             (0),
            .VALID_IN_EARLINESS             (0),
            .REGISTERED_DATA_OUT_COUNT      (DATA_WIDTH),
            .NEVER_OVERFLOWS                (0),
            .HOLD_DATA_OUT_WHEN_EMPTY       (0),
            .WRITE_AND_READ_DURING_FULL     (0),
            .RAM_BLOCK_TYPE                 ("FIFO_TO_CHOOSE"),
            .enable_ecc                     (enable_ecc)
          )
          hld_fifo_inst
          (
            .clock                          (clock),
            .resetn                         (resetn),
            .i_valid                        (input_accepted_for_fifo),
            .i_data                         (data_in_for_fifo),
            .o_stall                        (), //handled externally
            .o_almost_full                  (), //handled externally
            .o_valid                        (valid_out),
            .o_data                         (data_out),
            .i_stall                        (~output_accepted_for_fifo),
            .o_almost_empty                 (),
            .o_empty                        (empty),
            .ecc_err_status                 (ecc_err_status)
          );
    end 
    else 
    begin // default to "basic"
          hld_fifo #(
            .STYLE                          ("ms"), //acl_mid_speed_fifo
            .WIDTH                          (DATA_WIDTH),
            .DEPTH                          (DEPTH),
            .ALMOST_EMPTY_CUTOFF            (0),
            .ALMOST_FULL_CUTOFF             (DEPTH - ALMOST_FULL_VALUE),
            .INITIAL_OCCUPANCY              (0),
            .ASYNC_RESET                    (ASYNC_RESET),
            .SYNCHRONIZE_RESET              (SYNCHRONIZE_RESET),
            .RESET_EVERYTHING               (0),
            .RESET_EXTERNALLY_HELD          (1),
            .STALL_IN_EARLINESS             (0),
            .VALID_IN_EARLINESS             (0),
            .REGISTERED_DATA_OUT_COUNT      (DATA_WIDTH),
            .NEVER_OVERFLOWS                (0),
            .HOLD_DATA_OUT_WHEN_EMPTY       (0),
            .WRITE_AND_READ_DURING_FULL     (0),
            .USE_STALL_LATENCY_UPSTREAM     (0),
            .USE_STALL_LATENCY_DOWNSTREAM   (0),
            .RAM_BLOCK_TYPE                 ("FIFO_TO_CHOOSE"),
            .enable_ecc                     (enable_ecc)
          )
          hld_fifo_inst
          (
            .clock                          (clock),
            .resetn                         (resetn),
            .i_valid                        (valid_in),
            .i_data                         (data_in),
            .o_stall                        (stall_out),
            .o_almost_full                  (almost_full),
            .o_valid                        (valid_out),
            .o_data                         (data_out),
            .i_stall                        (stall_in),
            .o_almost_empty                 (),
            .o_empty                        (empty),
            .ecc_err_status                 (ecc_err_status)
          );
          assign full = stall_out;
          assign usedw = '0;
    end
  endgenerate


endmodule

`default_nettype wire
