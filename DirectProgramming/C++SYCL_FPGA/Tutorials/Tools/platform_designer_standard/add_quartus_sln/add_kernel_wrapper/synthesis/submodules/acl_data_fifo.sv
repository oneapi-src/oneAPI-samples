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
// protocol ports. Device implementation can be selected via parameters.
//
// DATA_WIDTH = 0:
//  Data width can be zero, in which case the the FIFO stores no data.
//
// Supported implementations (DATA_WIDTH > 0):
//  ram: RAM-based FIFO (min. latency 3)
//  ll_reg: low-latency register-based FIFO (min. latency 1)
//  ll_ram: low-latency RAM (min. latency 1; combination of ll_reg + ram)
//  zl_reg: zero-latency ll_reg (adds bypass path)
//  zl_ram: zero-latency ll_ram (adds bypass path)
//
// Supported implementations (DATA_WIDTH == 0);
//  For DATA_WIDTH == 0, the latency is either 1 ("low") or 0.
//  All implementations mentioned above are supported, with their implications
//  for either using the "ll_counter" or the "zl_counter"
//  (i.e. ram/ll_reg/ll_ram -> ll_counter, zl_reg/zl_ram -> zl_counter).
//
// STRICT_DEPTH:
//  A value of 0 means the FIFO that is instantiated will have a depth
//  of at least DEPTH. A value of 1 means the FIFO will have a depth exactly
//  equal to DEPTH.
//
//===----------------------------------------------------------------------===//

// altera message_off 10034

`default_nettype none

module acl_data_fifo 
#(
    parameter integer DATA_WIDTH = 32,          // >=0
    parameter integer DEPTH = 32,               // >0
    parameter integer STRICT_DEPTH = 0,         // 0|1 (1 == FIFO depth will be EXACTLY equal to DEPTH, otherwise >= DEPTH)
    parameter integer ALLOW_FULL_WRITE = 0,     // 0|1 (only supported by pure reg fifos: ll_reg, zl_reg, ll_counter, zl_counter)
    parameter integer INITIALIZE_TO_VALID = 0,  // Assume initial state is valid data.  Only allowed for depth-1 ll_reg.
    parameter INITIAL_FIFO_VALUE = 32'b0,  // Initial value to use, when INITIALIZE_TO_VALID is 1

    parameter string IMPL = "ram",              // see above (ram|ll_reg|ll_ram|zl_reg|zl_ram|ll_counter|zl_counter)
    parameter integer ALMOST_FULL_VALUE = 0,    // >= 0
    parameter LPM_HINT = "unused",
    parameter integer BACK_LL_REG_DEPTH = 2,  // the depth of the back ll_reg in sandwich impl, the default is 2
    parameter string ACL_FIFO_IMPL = "basic",   // impl: (basic|pow_of_2_full|pow_of_2_full_reg_data_in|pow_of_2_full_reg_output_accepted|pow_of_2_full_reg_data_in_reg_output_accepted)
    parameter bit ASYNC_RESET = 1,              // how do the registers CONSUME reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 0,         // before consumption, do we SYNCHRONIZE the reset: 1 means use a synchronizer (assume reset arrived asynchronously), 0 means passthrough (assume reset was already synchronized)
    parameter enable_ecc = "FALSE"               // Enable error correction coding
)
(
    input wire clock,
    input wire resetn,
    input wire [DATA_WIDTH-1:0] data_in,       // not used if DATA_WIDTH=0
    output logic [DATA_WIDTH-1:0] data_out,     // not used if DATA_WIDTH=0
    input wire valid_in,
    output logic valid_out,
    input wire stall_in,
    output logic stall_out,
    output logic  [1:0] ecc_err_status, // ecc status signals

    // internal signals (not required to use)
    output logic empty,
    output logic full,
    output logic almost_full
);
    
    //reset
    logic aclrn, sclrn, resetn_synchronized;
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
        .o_resetn_synchronized  (resetn_synchronized),
        .o_sclrn                (sclrn)
    );
    
    generate
        if( DATA_WIDTH > 0 )
        begin
            if( IMPL == "ram" )
            begin
                // Normal RAM FIFO.
                // Note that ALLOW_FULL_WRITE == 1 is not supported.
                acl_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(DEPTH),
                    .ALMOST_FULL_VALUE(ALMOST_FULL_VALUE),
                    .LPM_HINT(LPM_HINT),
                    .IMPL(ACL_FIFO_IMPL),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(data_in),
                    .data_out(data_out),
                    .valid_in(valid_in),
                    .valid_out(valid_out),
                    .stall_in(stall_in),
                    .stall_out(stall_out),
                    .empty(empty),
                    .full(full),
                    .almost_full(almost_full),
                    .ecc_err_status(ecc_err_status)
                );
            end
            else if( (IMPL == "ll_reg" || IMPL == "shift_reg") && DEPTH >= 2 && !ALLOW_FULL_WRITE )
            begin
                // For ll_reg's create an ll_fifo of DEPTH-1 with ALLOW_FULL_WRITE=1 followed by a staging register
                logic r_valid;
                logic [DATA_WIDTH-1:0] r_data;
                logic staging_reg_stall;

                localparam ALMOST_FULL_DEPTH_LOG2 = $clog2(DEPTH); // required to be DEPTH, this guarantees that almost_full=1 iff fifo occupancy >= ALMOST_FULL_VALUE
                localparam ALMOST_FULL_DEPTH_SNAPPED_TO_POW_OF_2 = 1 << ALMOST_FULL_DEPTH_LOG2;
                localparam ALMOST_FULL_COUNTER_OFFSET = ALMOST_FULL_DEPTH_SNAPPED_TO_POW_OF_2 - ALMOST_FULL_VALUE;

                logic [ALMOST_FULL_DEPTH_LOG2:0]  almost_full_counter;
                logic    input_accepted_for_counter;
                logic    output_accepted_for_counter;
                logic    sub_fifo_full;
                logic    sub_fifo_empty;
                
                assign  input_accepted_for_counter  = valid_in & ~stall_out;
                assign  output_accepted_for_counter = ~stall_in & valid_out;
                assign  almost_full                 = almost_full_counter[ALMOST_FULL_DEPTH_LOG2];
                // FIFO is full if the depth-1 fifo is full, and the staging reg has valid data
                assign  full                        = sub_fifo_full & staging_reg_stall;
                assign  empty                       = sub_fifo_empty & !staging_reg_stall;

                always @(posedge clock or negedge aclrn) begin
                  if (~aclrn) begin
                    almost_full_counter <= ALMOST_FULL_COUNTER_OFFSET;
                  end
                  else begin
                    almost_full_counter <= almost_full_counter  + input_accepted_for_counter - output_accepted_for_counter;
                    if (~sclrn) almost_full_counter <= ALMOST_FULL_COUNTER_OFFSET;
                  end
                end 

                acl_data_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(DEPTH-1),
                    .ALLOW_FULL_WRITE(1),
                    .IMPL(IMPL),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(data_in),
                    .data_out(r_data),
                    .valid_in(valid_in),
                    .valid_out(r_valid),
                    .empty(sub_fifo_empty),
                    .full(sub_fifo_full),
                    .stall_in(staging_reg_stall),
                    .stall_out(stall_out),
                    .ecc_err_status(ecc_err_status)
                );
                acl_staging_reg #(
                   .WIDTH(DATA_WIDTH),
                   .ASYNC_RESET(ASYNC_RESET),
                   .SYNCHRONIZE_RESET(0)
                ) staging_reg (
                   .clk(clock), 
                   .reset(~resetn_synchronized), 
                   .i_data(r_data), 
                   .i_valid(r_valid), 
                   .o_stall(staging_reg_stall), 
                   .o_data(data_out), 
                   .o_valid(valid_out), 
                   .i_stall(stall_in)
                );
            end
            else if( IMPL == "shift_reg" && DEPTH <= 1)
            begin
                // Depth-1 shift register
                // Case:149478 Removed individual no-shift-reg
                // assignments.

                logic [DEPTH-1:0] r_valid_NO_SHIFT_REG;
                logic [DATA_WIDTH-1:0] r_data_NO_SHIFT_REG;

                assign empty = 1'b0;
                assign full = 1'b1;

                always @(posedge clock or negedge aclrn) begin
                    if (!aclrn) begin
                        r_valid_NO_SHIFT_REG <= 1'b0;
                    end
                    else begin
                        if (!stall_in) begin
                            r_valid_NO_SHIFT_REG <= valid_in;
                        end
                        if (!sclrn) r_valid_NO_SHIFT_REG <= 1'b0;
                    end
                end

                always @(posedge clock) begin
                    if (!stall_in) begin
                        r_data_NO_SHIFT_REG <= data_in;
                    end
                end

                assign stall_out = stall_in;
                assign valid_out = r_valid_NO_SHIFT_REG;
                assign data_out = r_data_NO_SHIFT_REG;
                assign ecc_err_status = 2'h0;
            end
            else if( IMPL == "ll_reg" && ALLOW_FULL_WRITE && DEPTH <= 1)
            begin
                // Depth-1 FIFO

                logic [DEPTH-1:0] r_valid_NO_SHIFT_REG;
                logic [DATA_WIDTH-1:0] r_data_NO_SHIFT_REG;
                logic do_stall;

                // ll_reg is stall-enable cluster - i.e. only stall
                // once we see valid data.
                assign do_stall = (stall_in & r_valid_NO_SHIFT_REG);
                assign full = r_valid_NO_SHIFT_REG;
                assign empty = !full;

                always @(posedge clock or negedge aclrn) begin
                    if (!aclrn) begin
                        r_valid_NO_SHIFT_REG <= INITIALIZE_TO_VALID;
                    end
                    else begin
                        if (!do_stall) begin
                            r_valid_NO_SHIFT_REG <= valid_in;
                        end
                        if (!sclrn) r_valid_NO_SHIFT_REG <= INITIALIZE_TO_VALID;
                    end
                end    

                // We only need a reset on the data if INITIALIZE_TO_VALID is 1.
                if (INITIALIZE_TO_VALID) begin
                    always @(posedge clock or negedge aclrn) begin
                        if (!aclrn) begin
                            r_data_NO_SHIFT_REG <= INITIAL_FIFO_VALUE;
                        end
                        else begin
                            if (!do_stall) begin
                                r_data_NO_SHIFT_REG <= data_in;
                            end
                            if (!sclrn) r_data_NO_SHIFT_REG <= INITIAL_FIFO_VALUE;
                        end
                    end
                end else begin
                    always @(posedge clock) begin
                        if (!do_stall) begin
                            r_data_NO_SHIFT_REG <= data_in;
                        end
                    end
                end

                assign stall_out = do_stall;
                assign valid_out = r_valid_NO_SHIFT_REG;
                assign data_out = r_data_NO_SHIFT_REG;
                assign ecc_err_status = 2'h0;
            end
            else if( IMPL == "shift_reg" )
            begin
                // Shift register implementation of a FIFO

                logic [DEPTH-1:0] r_valid;
                logic [DATA_WIDTH-1:0] r_data[0:DEPTH-1];

                assign empty = 1'b0;

                always @(posedge clock or negedge aclrn) begin
                    if (!aclrn) begin
                        r_valid <= {(DEPTH){1'b0}};
                    end
                    else begin
                        if (!stall_in) begin
                            r_valid[0] <= valid_in;
                            for (int i = 1; i < DEPTH; i++) begin : GEN_RANDOME_BLOCK_NAME_R3
                                    r_valid[i] <= r_valid[i - 1];
                            end
                        end
                        if (!sclrn) r_valid <= {(DEPTH){1'b0}};
                    end
                end    

                always @(posedge clock) begin
                     if (!stall_in) begin
                         r_data[0]  <= data_in;
                         for (int i = 1; i < DEPTH; i++) begin : GEN_RANDOM_BLOCK_NAME_R4
                                 r_data[i]  <= r_data[i - 1];
                         end
                     end
                end
                assign stall_out = stall_in; 
                assign valid_out = r_valid[DEPTH-1];
                assign data_out = r_data[DEPTH-1];
                assign ecc_err_status = 2'h0;
            end
            else if( IMPL == "ll_reg" )
            begin
                // LL REG FIFO. Supports ALLOW_FULL_WRITE == 1.
                logic write, read;

                assign write = valid_in & ~stall_out;
                assign read = ~stall_in & ~empty;

                acl_ll_fifo #(
                    .WIDTH(DATA_WIDTH),
                    .DEPTH(DEPTH),
                    .ALMOST_FULL_VALUE(ALMOST_FULL_VALUE),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0)
                )
                fifo (
                    .clk(clock),
                    .reset(~resetn_synchronized),
                    .data_in(data_in),
                    .write(write),
                    .data_out(data_out),
                    .read(read),
                    .empty(empty),
                    .full(full),
                    .almost_full(almost_full)
                );

                assign valid_out = ~empty;
                assign stall_out = ALLOW_FULL_WRITE ? (full & stall_in) : full;
                assign ecc_err_status = 2'h0;
            end
            else if( IMPL == "ll_ram" )
            begin
                // LL RAM FIFO.
                acl_ll_ram_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(DEPTH),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(data_in),
                    .data_out(data_out),
                    .valid_in(valid_in),
                    .valid_out(valid_out),
                    .stall_in(stall_in),
                    .stall_out(stall_out),
                    .empty(empty),
                    .full(full),
                    .ecc_err_status(ecc_err_status)
                );
            end
            else if( IMPL == "passthrough" )
            begin
                // Useful for turning off a FIFO and making it into a wire
                assign valid_out = valid_in; 
                assign stall_out = stall_in;
                assign data_out = data_in;
                assign ecc_err_status = 2'h0;
            end
            else if( IMPL == "ram_plus_reg" )
            begin
                logic [DATA_WIDTH-1:0] rdata2;
                logic v2;
                logic s2;
                logic [1:0] ecc_err_status_0, ecc_err_status_1;
                acl_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(DEPTH),
                    .ALMOST_FULL_VALUE(ALMOST_FULL_VALUE),
                    .LPM_HINT(LPM_HINT),
                    .IMPL(ACL_FIFO_IMPL),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo_inner (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(data_in),
                    .data_out(rdata2),
                    .valid_in(valid_in),
                    .valid_out(v2),
                    .stall_in(s2),
                    .empty(empty),
                    .stall_out(stall_out),
                    .almost_full(almost_full),
                    .ecc_err_status(ecc_err_status_0)
                );
                acl_data_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(2),
                    .IMPL("ll_reg"),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo_outer (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(rdata2),
                    .data_out(data_out),
                    .valid_in(v2),
                    .valid_out(valid_out),
                    .stall_in(stall_in),
                    .stall_out(s2),
                    .ecc_err_status(ecc_err_status_1)
                );
                assign ecc_err_status = ecc_err_status_0 | ecc_err_status_1;
            end
            else if( IMPL == "sandwich" )
            begin
                logic [DATA_WIDTH-1:0] rdata1;
                logic [DATA_WIDTH-1:0] rdata2;
                logic v1, v2;
                logic s1, s2;
                logic [1:0] ecc_err_status_0, ecc_err_status_1, ecc_err_status_2;
                acl_data_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(2),
                    .IMPL("ll_reg"),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo_outer1 (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(data_in),
                    .data_out(rdata1),
                    .valid_in(valid_in),
                    .valid_out(v1),
                    .stall_in(s1),
                    .stall_out(stall_out),
                    .ecc_err_status(ecc_err_status_0)
                );
                acl_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(DEPTH),
                    .ALMOST_FULL_VALUE(ALMOST_FULL_VALUE),
                    .LPM_HINT(LPM_HINT),
                    .IMPL(ACL_FIFO_IMPL),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo_inner (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(rdata1),
                    .data_out(rdata2),
                    .valid_in(v1),
                    .valid_out(v2),
                    .stall_in(s2),
                    .stall_out(s1),
                    .empty(empty),
                    .almost_full(almost_full),
                    .ecc_err_status(ecc_err_status_1)
                );
                acl_data_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(BACK_LL_REG_DEPTH),
                    .IMPL("ll_reg"),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo_outer2 (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(rdata2),
                    .data_out(data_out),
                    .valid_in(v2),
                    .valid_out(valid_out),
                    .stall_in(stall_in),
                    .stall_out(s2),
                    .ecc_err_status(ecc_err_status_2)
                );
                assign ecc_err_status = ecc_err_status_0 | ecc_err_status_1 | ecc_err_status_2;
            end
            else if( IMPL == "zl_reg" || IMPL == "zl_ram" )
            begin
                // ZL RAM/REG FIFO.
                logic [DATA_WIDTH-1:0] fifo_data_in, fifo_data_out;
                logic fifo_valid_in, fifo_valid_out;
                logic fifo_stall_in, fifo_stall_out;

                acl_data_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(DEPTH),
                    .ALLOW_FULL_WRITE(ALLOW_FULL_WRITE),
                    .IMPL(IMPL == "zl_reg" ? "ll_reg" : "ll_ram"),
                    .ALMOST_FULL_VALUE(ALMOST_FULL_VALUE),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .data_in(fifo_data_in),
                    .data_out(fifo_data_out),
                    .valid_in(fifo_valid_in),
                    .valid_out(fifo_valid_out),
                    .stall_in(fifo_stall_in),
                    .stall_out(fifo_stall_out),
                    .empty(empty),
                    .full(full),
                    .almost_full(almost_full),
                    .ecc_err_status(ecc_err_status)
                );

		logic staging_reg_stall;

                assign fifo_data_in = data_in;
                assign fifo_valid_in = valid_in & (staging_reg_stall | fifo_valid_out);
                assign fifo_stall_in = staging_reg_stall;

                assign stall_out = fifo_stall_out;

                 // Staging register to break the stall path
                acl_staging_reg #(
                   .WIDTH(DATA_WIDTH),
                   .ASYNC_RESET(ASYNC_RESET),
                   .SYNCHRONIZE_RESET(0)
                ) staging_reg (
                   .clk(clock), 
                   .reset(~resetn_synchronized), 
                   .i_data(fifo_valid_out ? fifo_data_out : data_in), 
                   .i_valid(fifo_valid_out | valid_in), 
                   .o_stall(staging_reg_stall), 
                   .o_data(data_out), 
                   .o_valid(valid_out), 
                   .i_stall(stall_in)
                );
            end
            else 
            begin
              assign ecc_err_status = 2'h0;
            end
         end
         else // DATA_WIDTH == 0
         begin
            if( IMPL == "ram" || IMPL == "ram_plus_reg" || IMPL == "ll_reg" || IMPL == "ll_ram" || IMPL == "ll_counter" )
            begin
                // LL counter fifo.
                acl_valid_fifo_counter #(
                    .DEPTH(DEPTH),
                    .STRICT_DEPTH(STRICT_DEPTH),
                    .ALLOW_FULL_WRITE(ALLOW_FULL_WRITE),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0)
                )
                counter (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .valid_in(valid_in),
                    .valid_out(valid_out),
                    .stall_in(stall_in),
                    .stall_out(stall_out),
                    .empty(empty),
                    .full(full)
                );
                assign ecc_err_status = 2'h0;
             end
             else if( IMPL == "zl_reg" || IMPL == "zl_ram" || IMPL == "zl_counter" )
             begin
                // ZL counter fifo.
                logic fifo_valid_in, fifo_valid_out;
                logic fifo_stall_in, fifo_stall_out;

                acl_data_fifo #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .DEPTH(DEPTH),
                    .STRICT_DEPTH(STRICT_DEPTH),
                    .ALLOW_FULL_WRITE(ALLOW_FULL_WRITE),
                    .IMPL("ll_counter"),
                    .ASYNC_RESET(ASYNC_RESET),
                    .SYNCHRONIZE_RESET(0),
                    .enable_ecc(enable_ecc)
                )
                fifo (
                    .clock(clock),
                    .resetn(resetn_synchronized),
                    .valid_in(fifo_valid_in),
                    .valid_out(fifo_valid_out),
                    .stall_in(fifo_stall_in),
                    .stall_out(fifo_stall_out),
                    .empty(empty),
                    .full(full),
                    .ecc_err_status(ecc_err_status)
                );

                assign fifo_valid_in = valid_in & (stall_in | fifo_valid_out);
                assign fifo_stall_in = stall_in;

                assign stall_out = fifo_stall_out;
                assign valid_out = fifo_valid_out | valid_in;
             end
         end
    endgenerate
endmodule

`default_nettype wire
