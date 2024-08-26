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


// If one can tolerate an extra clock cycle of latency from the increment signal to when the counter is updated
// then it gives sufficient time to do lookahead (bottom bits are 1) so that we can generate independent 
// increment signals for each bit range. Without the extra latency this strategy can still be done, but then
// we would need combinational logic for the clock enables for each bit range of the counter.

// This module was designed in conjunction with acl_high_speed_fifo.sv. The testbench is grouped together with the fifo.

// Required files:
// -acl_tessellated_incr_lookahead.sv
// -acl_reset_handler.sv
// -acl_fanout_pipeline.sv
// -acl_std_synchronizer_nocut.sv

// TEMPORARY FIX:
// Reset values are set to match the behavior of before reset cleanup. See FB case:457213. This is a workaround for now.
// Eventually the compiler needs to set the reset parameters correctly, at which point the default values will be set
// back to the original intent, which is for someone who knows nothing about the reset in their system.


`default_nettype none

module acl_tessellated_incr_lookahead #(
    //general configuration
    parameter int WIDTH,                        // width of the counter, no limit but no longer logic depth 1 at 15 or higher, and also INITIAL_OCCUPANCY is inherently only 32 bits
    parameter int INITIAL_OCCUPANCY = 0,        // initial value of the counter
    
    //reset configuration
    parameter bit ASYNC_RESET = 1,              // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 0,        // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    parameter bit RESET_EVERYTHING = 0,         // intended for partial reconfig debug, set to 1 to reset every register (normally async reset excludes data path and sync reset additionally excludes some control signals)
    parameter bit RESET_EXTERNALLY_HELD = 1     // set to 1 if resetn will be held for at least THREE clock cycles, otherwise we will internally pulse stretch reset before consumption
) (
    input  wire                 clock,
    input  wire                 resetn,         // if using sync reset, resetn must be held for at least THREE clocks (cnt -> lo_ones -> incr_stage), otherwise set RESET_EXTERNALLY_HELD=0 so pulse stretch is dnoe internally
    input  wire                 incr,           // increment the count as of 2 clock cycles later
    output logic    [WIDTH-1:0] count           // count that tracks increment, resets to 0
);
    
    // basic functionality of this circuit
    // logic incr_prev;
    // always_ff @(posedge clock) begin
    //     incr_prev <= incr;
    //     count <= count + incr_prev;
    //     if (~resetn) begin
    //         incr_prev <= 1'b0;
    //         count <= INITIAL_OCCUPANCY;
    //     end
    // end
    
    //reset
    logic aclrn, sclrn;
    logic sclrn_before_pulse_stretch, sclrn_after_pulse_stretch, sclrn_base;
    logic [2:0] sclrn_chain;
    
    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
        .PIPE_DEPTH             (0),
        .NUM_COPIES             (1)
    )
    acl_reset_handler_inst
    (
        .clk                    (clock),
        .i_resetn               (resetn),
        .o_aclrn                (aclrn),
        .o_resetn_synchronized  (),
        .o_sclrn                (sclrn_before_pulse_stretch)
    );
    
    generate
    if (RESET_EXTERNALLY_HELD) begin : gen_direct_sclrn
        assign sclrn = sclrn_before_pulse_stretch;
    end
    else begin : gen_pulse_stretch_sclrn
        always_ff @(posedge clock) begin
            sclrn_chain <= (sclrn_chain << 1) | sclrn_before_pulse_stretch;
            sclrn_after_pulse_stretch <= &sclrn_chain;
            sclrn_base <= sclrn_after_pulse_stretch;
            sclrn <= sclrn_base;
        end
    end
    endgenerate
    
    
    // tessellation configuration:
    //  - only up to 4 stages are allowed because lower 2 bits can roll over before fifth stage can update
    //  - for 6-lut, need to be at most WIDTH = 14, beyond this the circuit will still work but no longer logic depth 1
    // 
    //  WIDTH   | FIRST     | SECOND    | THIRD     | FOURTH
    //  --------+-----------+-----------+-----------+-------
    //  1       | 1         | 0         | 0         | 0
    //  2       | 2         | 0         | 0         | 0
    //  3       | 3         | 0         | 0         | 0
    //  4       | 4         | 0         | 0         | 0
    //  --------+-----------+-----------+-----------+-------
    //  5       | 2         | 3         | 0         | 0
    //  6       | 2         | 4         | 0         | 0
    //  --------+-----------+-----------+-----------+-------
    //  7       | 2         | 2         | 3         | 0
    //  8       | 2         | 3         | 3         | 0
    //  9       | 2         | 3         | 4         | 0
    //  10      | 2         | 4         | 4         | 0
    //  --------+-----------+-----------+-----------+-------
    //  11      | 2         | 3         | 3         | 3
    //  12      | 2         | 3         | 3         | 4
    //  13      | 2         | 3         | 4         | 4
    //  14      | 2         | 4         | 4         | 4
    //  --------+-----------+-----------+-----------+-------
    //  15      | 2         | 4         | 4         | 5
    //  16      | 2         | 4         | 5         | 5
    //  17      | 2         | 5         | 5         | 5
    
    localparam bit [31:0] INIT_OCC = INITIAL_OCCUPANCY;   //allows for bit selects into INIT_OCC
    localparam int STAGES = (WIDTH<=4) ? 1 : (WIDTH<=6) ? 2 : (WIDTH<=10) ? 3 : 4;
    localparam int UPPER_STAGES = (STAGES<=2) ? 1 : (STAGES-1);
    localparam int UPPER_WIDTH = (WIDTH<=4) ? 4 : (WIDTH-2);
    
    localparam int FIRST_BITS = (STAGES==1) ? WIDTH : 2;
    localparam int SECOND_BITS = (STAGES==1) ? 0 : (UPPER_WIDTH/UPPER_STAGES);
    localparam int THIRD_BITS = (STAGES<=2) ? 0 : ((2*UPPER_WIDTH/UPPER_STAGES)-(UPPER_WIDTH/UPPER_STAGES));
    localparam int FOURTH_BITS = (STAGES<=3) ? 0 : ((3*UPPER_WIDTH/UPPER_STAGES)-(2*UPPER_WIDTH/UPPER_STAGES));
    
    //first stage
    logic first_incr;
    logic [FIRST_BITS-1:0] first_cnt;
    
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            first_incr <= 1'b0;
            first_cnt <= INIT_OCC[0 +: FIRST_BITS];
        end
        else begin
            first_incr <= incr;
            first_cnt <= first_cnt + first_incr;
            if (~sclrn) begin
                first_incr <= 1'b0;
                first_cnt <= INIT_OCC[0 +: FIRST_BITS];
            end
        end
    end
    
    //second stage
    logic first_ones, second_incr;
    logic [SECOND_BITS-1:0] second_cnt;
    generate
    if (SECOND_BITS) begin
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                first_ones <= (INIT_OCC[0 +: FIRST_BITS] == {FIRST_BITS{1'h1}}) ? 1'b1 : 1'b0;
                second_incr <= 1'b0;
                second_cnt <= INIT_OCC[FIRST_BITS +: SECOND_BITS];
            end
            else begin
                first_ones <= ((incr & first_incr) ? (first_cnt==2'h1) : (incr | first_incr) ? (first_cnt==2'h2) : (first_cnt==2'h3)) ? 1'b1 : 1'b0;
                second_incr <= incr & first_ones;
                second_cnt <= second_cnt + second_incr;
                if (~sclrn) begin
                    if (RESET_EVERYTHING) first_ones <= (INIT_OCC[0 +: FIRST_BITS] == {FIRST_BITS{1'h1}}) ? 1'b1 : 1'b0;
                    second_incr <= 1'b0;
                    second_cnt <= INIT_OCC[FIRST_BITS +: SECOND_BITS];
                end
            end
        end
    end
    endgenerate
    
    //third stage
    logic second_ones, third_incr;
    logic [THIRD_BITS-1:0] third_cnt;
    generate
    if (THIRD_BITS) begin
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                second_ones <= (INIT_OCC[FIRST_BITS +: SECOND_BITS] == {SECOND_BITS{1'h1}}) ? 1'b1 : 1'b0;
                third_incr <= 1'b0;
                third_cnt <= INIT_OCC[(FIRST_BITS+SECOND_BITS) +: THIRD_BITS];
            end
            else begin
                second_ones <= &second_cnt;
                third_incr <= incr & first_ones & second_ones;
                third_cnt <= third_cnt + third_incr;
                if (~sclrn) begin
                    if (RESET_EVERYTHING) second_ones <= (INIT_OCC[FIRST_BITS +: SECOND_BITS] == {SECOND_BITS{1'h1}}) ? 1'b1 : 1'b0;
                    third_incr <= 1'b0;
                    third_cnt <= INIT_OCC[(FIRST_BITS+SECOND_BITS) +: THIRD_BITS];
                end
            end
        end
    end
    endgenerate
    
    //fourth stage
    logic third_ones, fourth_incr;
    logic [FOURTH_BITS-1:0] fourth_cnt;
    generate
    if (FOURTH_BITS) begin
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                third_ones <= (INIT_OCC[(FIRST_BITS+SECOND_BITS) +: THIRD_BITS] == {THIRD_BITS{1'h1}}) ? 1'b1 : 1'b0;
                fourth_incr <= 1'b0;
                fourth_cnt <= INIT_OCC[(FIRST_BITS+SECOND_BITS+THIRD_BITS) +: FOURTH_BITS];
            end
            else begin
                third_ones <= &third_cnt;
                fourth_incr <= incr & first_ones & second_ones & third_ones;
                fourth_cnt <= fourth_cnt + fourth_incr;
                if (~sclrn) begin
                    if (RESET_EVERYTHING) third_ones <= (INIT_OCC[(FIRST_BITS+SECOND_BITS) +: THIRD_BITS] == {THIRD_BITS{1'h1}}) ? 1'b1 : 1'b0;
                    fourth_incr <= 1'b0;
                    fourth_cnt <= INIT_OCC[(FIRST_BITS+SECOND_BITS+THIRD_BITS) +: FOURTH_BITS];
                end
            end
        end
    end
    endgenerate
    
    //output
    generate
    if (STAGES == 1) begin
        assign count = first_cnt;
    end
    if (STAGES == 2) begin
        assign count = {second_cnt, first_cnt};
    end
    if (STAGES == 3) begin
        assign count = {third_cnt, second_cnt, first_cnt};
    end
    if (STAGES == 4) begin
        assign count = {fourth_cnt, third_cnt, second_cnt, first_cnt};
    end
    endgenerate
    
endmodule



`default_nettype wire
