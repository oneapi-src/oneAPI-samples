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


// Linear feedback shift register, maximal length sequences using only 2-input xor gates. Use Fibonacci LFSR if WIDTH has a known configuration 
// using only 2 taps in the single feedback xor. Otherwise use Galois LFSR (2-input xor gates inside the shift register, only 3 are needed)
//
// This implementation uses XNOR, which means the illegal state is all ones. With initial occupancy as 0, state will reset to 0.
//
// IMPORTANT: this module has only been tested up to 24 bits. The LFSR tables have been extracted from "Table of Linear Feedback Shift Registers"
// by Roy Wardand and Tim Molteno, published in 2007. There is probably no easy way to validate that a sequence goes through all 2^N-1 states
// exactly once as N gets large.
//
// This module was designed in conjunction with acl_high_speed_fifo.sv. The testbench is grouped together with the fifo.
//
// Required files:
// -acl_lfsr.sv
// -acl_reset_handler.sv
//
// TEMPORARY FIX:
// Reset values are set to match the behavior of before reset cleanup. See FB case:457213. This is a workaround for now.
// Eventually the compiler needs to set the reset parameters correctly, at which point the default values will be set
// back to the original intent, which is for someone who knows nothing about the reset in their system.

`default_nettype none

module acl_lfsr #(
    parameter int WIDTH,                // at least 2, up to 512
    parameter bit ASYNC_RESET=1,        // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET=0,  // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    //RESET_EVERYTHING not needed since we already reset all registers
    //RESET_EXTERNALLY_HELD not needed since reset pulse stretcher is not needed
    parameter int INITIAL_OCCUPANCY=0   // if enable was asserted for INITIAL_OCCUPANCY clock cycles what would the state of the lfsr be, use this as the reset value instead of the original reset value
)
(
    input  wire                 clock,
    input  wire                 resetn,    // WARNING: to enable retiming one must set ASYNC_RESET=0 and provide a register chain driving this resetn outside of this module
    input  wire                 enable,
    output logic    [WIDTH-1:0] state
);

logic aclrn, sclrn;
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
    .o_sclrn                (sclrn)
);

generate
if (WIDTH == 2) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  1),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 3) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  2),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 4) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  3),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 5) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  3),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 6) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  5),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 7) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  6),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 8) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(  6), .T2(  5), .T3(  4), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 9) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  5),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 10) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  7),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 11) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(  9),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 12) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 11), .T2(  8), .T3(  6), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 13) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 12), .T2( 10), .T3(  9), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 14) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 13), .T2( 11), .T3(  9), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 15) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 14),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 16) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 14), .T2( 13), .T3( 11), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 17) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 14),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 18) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 11),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 19) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 18), .T2( 17), .T3( 14), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 20) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 17),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 21) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 19),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 22) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 21),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 23) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 18),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 24) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 23), .T2( 21), .T3( 20), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 25) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 22),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 26) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 25), .T2( 24), .T3( 20), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 27) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 26), .T2( 25), .T3( 22), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 28) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 25),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 29) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 27),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 30) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 29), .T2( 26), .T3( 24), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 31) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 28),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 32) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 30), .T2( 26), .T3( 25), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 33) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 20),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 34) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 31), .T2( 30), .T3( 26), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 35) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 33),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 36) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 25),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 37) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 36), .T2( 33), .T3( 31), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 38) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 37), .T2( 33), .T3( 32), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 39) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 35),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 40) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 37), .T2( 36), .T3( 35), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 41) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 38),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 42) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 40), .T2( 37), .T3( 35), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 43) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 42), .T2( 38), .T3( 37), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 44) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 42), .T2( 39), .T3( 38), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 45) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 44), .T2( 42), .T3( 41), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 46) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 40), .T2( 39), .T3( 38), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 47) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 42),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 48) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 44), .T2( 41), .T3( 39), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 49) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 40),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 50) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 48), .T2( 47), .T3( 46), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 51) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 50), .T2( 48), .T3( 45), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 52) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 49),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 53) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 52), .T2( 51), .T3( 47), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 54) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 51), .T2( 48), .T3( 46), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 55) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 31),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 56) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 54), .T2( 52), .T3( 49), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 57) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 50),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 58) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 39),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 59) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 57), .T2( 55), .T3( 52), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 60) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 59),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 61) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 60), .T2( 59), .T3( 56), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 62) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 59), .T2( 57), .T3( 56), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 63) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 62),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 64) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 63), .T2( 61), .T3( 60), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 65) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 47),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 66) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 60), .T2( 58), .T3( 57), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 67) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 66), .T2( 65), .T3( 62), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 68) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 59),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 69) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 67), .T2( 64), .T3( 63), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 70) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 69), .T2( 67), .T3( 65), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 71) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 65),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 72) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 69), .T2( 63), .T3( 62), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 73) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 48),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 74) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 71), .T2( 70), .T3( 67), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 75) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 74), .T2( 72), .T3( 69), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 76) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 74), .T2( 72), .T3( 71), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 77) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 75), .T2( 72), .T3( 71), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 78) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 77), .T2( 76), .T3( 71), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 79) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 70),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 80) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 78), .T2( 76), .T3( 71), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 81) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 77),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 82) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 78), .T2( 76), .T3( 73), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 83) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 81), .T2( 79), .T3( 76), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 84) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 71),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 85) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 84), .T2( 83), .T3( 77), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 86) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 84), .T2( 81), .T3( 80), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 87) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 74),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 88) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 80), .T2( 79), .T3( 77), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 89) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 51),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 90) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 88), .T2( 87), .T3( 85), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 91) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 90), .T2( 86), .T3( 83), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 92) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 90), .T2( 87), .T3( 86), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 93) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 91),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 94) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 73),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 95) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 84),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 96) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 90), .T2( 87), .T3( 86), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 97) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 91),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 98) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 87),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 99) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 95), .T2( 94), .T3( 92), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 100) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 63),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 101) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(100), .T2( 95), .T3( 94), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 102) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1( 99), .T2( 97), .T3( 96), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 103) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 94),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 104) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(103), .T2( 94), .T3( 93), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 105) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 89),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 106) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 91),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 107) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(105), .T2( 99), .T3( 98), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 108) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 77),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 109) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(107), .T2(105), .T3( 10), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 110) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(109), .T2(106), .T3( 10), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 111) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(101),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 112) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(108), .T2(106), .T3( 10), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 113) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(104),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 114) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(113), .T2(112), .T3( 10), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 115) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(110), .T2(108), .T3( 10), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 116) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(114), .T2(111), .T3( 11), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 117) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(116), .T2(115), .T3( 11), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 118) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 85),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 119) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(111),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 120) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(118), .T2(114), .T3(111), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 121) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(103),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 122) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(121), .T2(120), .T3(116), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 123) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(121),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 124) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 87),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 125) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(120), .T2(119), .T3(118), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 126) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(124), .T2(122), .T3(119), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 127) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(126),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 128) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(127), .T2(126), .T3(121), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 129) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(124),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 130) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(127),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 131) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(129), .T2(128), .T3(123), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 132) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(103),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 133) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(131), .T2(125), .T3(124), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 134) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 77),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 135) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(124),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 136) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(134), .T2(133), .T3(128), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 137) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(116),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 138) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(137), .T2(131), .T3(130), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 139) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(136), .T2(134), .T3(131), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 140) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(111),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 141) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(140), .T2(135), .T3(128), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 142) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(121),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 143) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(141), .T2(140), .T3(138), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 144) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(142), .T2(140), .T3(137), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 145) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 93),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 146) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(144), .T2(143), .T3(141), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 147) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(145), .T2(143), .T3(136), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 148) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(121),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 149) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(142), .T2(140), .T3(139), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 150) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 97),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 151) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(148),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 152) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(150), .T2(149), .T3(146), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 153) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(152),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 154) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(153), .T2(149), .T3(145), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 155) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(151), .T2(150), .T3(148), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 156) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(153), .T2(151), .T3(147), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 157) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(155), .T2(152), .T3(151), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 158) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(153), .T2(152), .T3(150), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 159) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(128),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 160) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(158), .T2(157), .T3(155), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 161) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(143),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 162) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(158), .T2(155), .T3(154), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 163) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(160), .T2(157), .T3(156), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 164) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(159), .T2(158), .T3(152), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 165) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(162), .T2(157), .T3(156), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 166) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(164), .T2(163), .T3(156), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 167) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(161),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 168) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(162), .T2(159), .T3(152), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 169) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(135),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 170) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(147),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 171) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(169), .T2(166), .T3(165), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 172) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(165),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 173) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(171), .T2(168), .T3(165), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 174) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(161),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 175) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(169),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 176) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(167), .T2(165), .T3(164), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 177) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(169),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 178) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1( 91),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 179) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(178), .T2(177), .T3(175), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 180) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(173), .T2(170), .T3(168), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 181) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(180), .T2(175), .T3(174), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 182) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(181), .T2(176), .T3(174), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 183) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(127),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 184) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(177), .T2(176), .T3(175), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 185) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(161),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 186) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(180), .T2(178), .T3(177), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 187) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(182), .T2(181), .T3(180), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 188) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(186), .T2(183), .T3(182), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 189) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(187), .T2(184), .T3(183), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 190) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(188), .T2(184), .T3(177), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 191) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(182),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 192) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(190), .T2(178), .T3(177), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 193) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(178),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 194) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(107),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 195) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(193), .T2(192), .T3(187), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 196) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(194), .T2(187), .T3(185), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 197) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(195), .T2(193), .T3(188), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 198) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(133),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 199) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(165),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 200) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(198), .T2(197), .T3(195), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 201) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(187),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 202) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(147),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 203) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(202), .T2(196), .T3(195), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 204) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(201), .T2(200), .T3(194), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 205) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(203), .T2(200), .T3(196), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 206) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(201), .T2(197), .T3(196), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 207) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(164),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 208) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(207), .T2(205), .T3(199), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 209) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(203),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 210) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(207), .T2(206), .T3(198), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 211) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(203), .T2(201), .T3(200), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 212) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(107),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 213) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(211), .T2(208), .T3(207), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 214) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(213), .T2(211), .T3(209), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 215) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(192),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 216) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(215), .T2(213), .T3(209), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 217) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(172),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 218) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(207),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 219) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(218), .T2(215), .T3(211), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 220) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(211), .T2(210), .T3(208), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 221) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(219), .T2(215), .T3(213), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 222) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(220), .T2(217), .T3(214), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 223) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(190),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 224) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(222), .T2(217), .T3(212), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 225) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(193),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 226) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(223), .T2(219), .T3(216), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 227) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(223), .T2(218), .T3(217), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 228) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(226), .T2(217), .T3(216), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 229) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(228), .T2(225), .T3(219), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 230) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(224), .T2(223), .T3(222), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 231) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(205),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 232) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(228), .T2(223), .T3(221), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 233) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(159),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 234) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(203),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 235) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(234), .T2(229), .T3(226), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 236) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(231),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 237) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(236), .T2(233), .T3(230), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 238) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(237), .T2(236), .T3(233), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 239) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(203),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 240) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(237), .T2(235), .T3(232), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 241) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(171),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 242) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(241), .T2(236), .T3(231), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 243) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(242), .T2(238), .T3(235), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 244) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(243), .T2(240), .T3(235), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 245) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(244), .T2(241), .T3(239), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 246) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(245), .T2(244), .T3(235), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 247) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(165),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 248) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(238), .T2(234), .T3(233), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 249) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(163),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 250) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(147),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 251) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(249), .T2(247), .T3(244), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 252) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(185),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 253) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(252), .T2(247), .T3(246), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 254) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(253), .T2(252), .T3(247), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 255) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(203),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 256) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(254), .T2(251), .T3(246), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 257) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(245),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 258) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(175),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 259) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(257), .T2(253), .T3(249), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 260) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(253), .T2(252), .T3(250), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 261) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(257), .T2(255), .T3(254), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 262) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(258), .T2(254), .T3(253), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 263) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(170),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 264) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(263), .T2(255), .T3(254), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 265) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(223),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 266) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(219),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 267) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(264), .T2(261), .T3(259), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 268) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(243),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 269) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(268), .T2(263), .T3(262), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 270) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(217),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 271) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(213),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 272) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(270), .T2(266), .T3(263), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 273) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(250),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 274) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(207),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 275) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(266), .T2(265), .T3(264), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 276) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(275), .T2(273), .T3(270), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 277) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(274), .T2(271), .T3(265), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 278) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(273),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 279) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(274),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 280) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(278), .T2(275), .T3(271), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 281) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(188),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 282) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(247),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 283) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(278), .T2(276), .T3(271), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 284) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(165),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 285) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(280), .T2(278), .T3(275), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 286) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(217),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 287) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(216),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 288) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(287), .T2(278), .T3(277), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 289) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(268),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 290) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(288), .T2(287), .T3(285), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 291) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(286), .T2(280), .T3(279), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 292) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(195),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 293) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(292), .T2(287), .T3(282), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 294) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(233),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 295) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(247),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 296) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(292), .T2(287), .T3(285), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 297) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(292),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 298) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(294), .T2(290), .T3(287), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 299) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(295), .T2(293), .T3(288), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 300) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(293),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 301) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(299), .T2(296), .T3(292), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 302) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(261),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 303) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(297), .T2(291), .T3(290), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 304) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(303), .T2(302), .T3(293), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 305) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(203),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 306) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(305), .T2(303), .T3(299), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 307) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(305), .T2(303), .T3(299), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 308) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(306), .T2(299), .T3(293), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 309) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(307), .T2(302), .T3(299), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 310) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(309), .T2(305), .T3(302), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 311) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(308), .T2(306), .T3(304), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 312) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(307), .T2(302), .T3(301), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 313) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(234),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 314) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(299),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 315) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(314), .T2(306), .T3(305), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 316) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(181),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 317) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(315), .T2(313), .T3(310), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 318) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(313), .T2(312), .T3(310), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 319) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(283),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 320) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(319), .T2(317), .T3(316), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 321) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(290),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 322) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(255),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 323) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(322), .T2(320), .T3(313), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 324) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(321), .T2(320), .T3(318), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 325) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(323), .T2(320), .T3(315), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 326) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(325), .T2(323), .T3(316), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 327) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(293),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 328) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(323), .T2(321), .T3(319), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 329) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(279),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 330) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(328), .T2(323), .T3(322), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 331) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(329), .T2(325), .T3(321), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 332) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(209),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 333) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(331),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 334) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(333), .T2(330), .T3(327), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 335) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(333), .T2(328), .T3(325), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 336) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(335), .T2(332), .T3(329), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 337) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(282),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 338) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(336), .T2(335), .T3(332), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 339) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(332), .T2(329), .T3(323), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 340) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(337), .T2(336), .T3(329), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 341) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(336), .T2(330), .T3(327), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 342) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(217),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 343) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(268),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 344) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(338), .T2(334), .T3(333), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 345) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(323),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 346) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(344), .T2(339), .T3(335), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 347) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(344), .T2(337), .T3(336), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 348) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(344), .T2(341), .T3(340), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 349) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(347), .T2(344), .T3(343), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 350) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(297),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 351) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(317),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 352) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(346), .T2(341), .T3(339), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 353) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(284),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 354) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(349), .T2(341), .T3(340), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 355) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(354), .T2(350), .T3(349), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 356) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(349), .T2(347), .T3(346), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 357) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(355), .T2(347), .T3(346), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 358) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(351), .T2(350), .T3(344), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 359) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(291),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 360) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(359), .T2(335), .T3(334), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 361) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(360), .T2(357), .T3(354), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 362) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(299),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 363) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(362), .T2(356), .T3(355), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 364) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(297),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 365) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(360), .T2(359), .T3(356), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 366) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(337),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 367) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(346),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 368) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(361), .T2(359), .T3(351), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 369) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(278),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 370) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(231),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 371) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(369), .T2(368), .T3(363), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 372) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(369), .T2(365), .T3(357), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 373) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(371), .T2(366), .T3(365), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 374) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(369), .T2(368), .T3(366), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 375) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(359),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 376) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(371), .T2(369), .T3(368), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 377) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(336),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 378) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(335),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 379) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(375), .T2(370), .T3(369), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 380) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(333),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 381) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(380), .T2(379), .T3(376), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 382) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(301),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 383) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(293),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 384) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(378), .T2(369), .T3(368), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 385) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(379),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 386) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(303),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 387) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(385), .T2(379), .T3(378), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 388) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(387), .T2(385), .T3(374), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 389) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(384), .T2(380), .T3(379), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 390) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(301),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 391) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(363),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 392) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(386), .T2(382), .T3(379), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 393) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(386),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 394) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(259),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 395) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(390), .T2(389), .T3(384), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 396) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(371),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 397) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(392), .T2(387), .T3(385), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 398) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(393), .T2(392), .T3(384), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 399) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(313),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 400) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(398), .T2(397), .T3(395), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 401) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(249),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 402) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(399), .T2(398), .T3(393), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 403) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(398), .T2(395), .T3(394), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 404) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(215),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 405) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(398), .T2(397), .T3(388), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 406) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(249),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 407) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(336),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 408) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(407), .T2(403), .T3(401), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 409) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(322),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 410) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(407), .T2(406), .T3(400), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 411) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(408), .T2(401), .T3(399), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 412) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(265),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 413) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(407), .T2(406), .T3(403), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 414) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(405), .T2(401), .T3(398), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 415) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(313),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 416) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(414), .T2(411), .T3(407), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 417) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(310),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 418) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(417), .T2(415), .T3(403), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 419) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(415), .T2(414), .T3(404), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 420) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(412), .T2(410), .T3(407), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 421) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(419), .T2(417), .T3(416), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 422) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(273),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 423) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(398),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 424) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(422), .T2(417), .T3(415), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 425) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(413),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 426) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(415), .T2(414), .T3(412), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 427) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(422), .T2(421), .T3(416), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 428) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(323),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 429) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(422), .T2(421), .T3(419), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 430) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(419), .T2(417), .T3(415), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 431) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(311),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 432) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(429), .T2(428), .T3(419), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 433) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(400),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 434) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(429), .T2(423), .T3(422), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 435) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(430), .T2(426), .T3(423), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 436) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(271),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 437) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(436), .T2(435), .T3(431), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 438) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(373),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 439) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(390),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 440) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(439), .T2(437), .T3(436), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 441) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(410),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 442) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(440), .T2(437), .T3(435), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 443) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(442), .T2(437), .T3(433), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 444) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(435), .T2(432), .T3(431), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 445) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(441), .T2(439), .T3(438), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 446) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(341),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 447) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(374),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 448) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(444), .T2(442), .T3(437), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 449) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(315),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 450) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(371),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 451) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(450), .T2(441), .T3(435), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 452) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(448), .T2(447), .T3(446), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 453) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(449), .T2(447), .T3(438), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 454) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(449), .T2(445), .T3(444), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 455) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(417),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 456) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(454), .T2(445), .T3(433), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 457) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(441),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 458) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(255),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 459) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(457), .T2(454), .T3(447), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 460) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(399),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 461) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(460), .T2(455), .T3(454), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 462) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(389),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 463) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(370),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 464) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(460), .T2(455), .T3(441), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 465) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(406),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 466) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(460), .T2(455), .T3(452), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 467) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(466), .T2(461), .T3(456), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 468) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(464), .T2(459), .T3(453), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 469) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(467), .T2(464), .T3(460), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 470) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(321),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 471) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(470),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 472) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(470), .T2(469), .T3(461), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 473) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(470), .T2(467), .T3(465), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 474) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(283),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 475) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(471), .T2(467), .T3(466), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 476) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(461),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 477) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(470), .T2(462), .T3(461), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 478) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(357),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 479) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(375),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 480) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(473), .T2(467), .T3(464), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 481) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(343),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 482) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(477), .T2(476), .T3(473), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 483) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(479), .T2(477), .T3(474), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 484) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(379),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 485) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(479), .T2(469), .T3(468), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 486) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(481), .T2(478), .T3(472), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 487) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(393),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 488) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(487), .T2(485), .T3(484), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 489) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(406),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 490) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(271),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 491) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(488), .T2(485), .T3(480), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 492) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(491), .T2(485), .T3(484), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 493) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(490), .T2(488), .T3(483), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 494) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(357),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 495) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(419),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 496) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(494), .T2(491), .T3(480), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 497) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(419),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 498) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(495), .T2(489), .T3(487), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 499) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(494), .T2(493), .T3(488), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 500) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(499), .T2(494), .T3(490), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 501) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(499), .T2(497), .T3(496), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 502) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(498), .T2(497), .T3(494), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 503) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(500),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 504) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(502), .T2(490), .T3(483), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 505) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(349),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 506) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(411),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 507) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(504), .T2(501), .T3(494), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 508) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(399),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 509) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(506), .T2(502), .T3(501), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 510) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(501), .T2(500), .T3(498), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 511) begin
    fibonacci_lfsr #(.WIDTH(WIDTH), .T1(501),                     .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
if (WIDTH == 512) begin
    galois_lfsr    #(.WIDTH(WIDTH), .T1(510), .T2(507), .T3(504), .INITIAL_OCCUPANCY(INITIAL_OCCUPANCY)) lfsr_inst (.clock(clock), .aclrn(aclrn), .sclrn(sclrn), .enable(enable), .state(state));
end
endgenerate

endmodule



//The high speed FIFO uses LFSRs as the write and read addresses to memory. In order for the FIFO to exit reset with
//a nonzero occupancy, we must be able to advance the write address forward INITIAL_OCCUPANCY times. Each of the LFSR
//implementations contain a function which advances the state exactly the same way as the hardware logic gates would
//when the LFSR is actually in use, but the function applies the state advancement to some temporary signals which
//ultimately set the parameter RESET_VALUE. The function has no logic cost, RESET_VALUE is a compile time constant.


module galois_lfsr #(
    parameter int WIDTH,
    parameter int T1,
    parameter int T2,
    parameter int T3,
    parameter int INITIAL_OCCUPANCY=0
)
(
    input  wire                 clock,
    input  wire                 aclrn,
    input  wire                 sclrn,
    input  wire                 enable,
    output logic    [WIDTH-1:0] state
);
function [WIDTH-1:0] galois_lfsr_reset_value;
input int T1, T2, T3, INITIAL_OCCUPANCY;
begin
    bit [WIDTH-1:0] old_state, new_state;
    int INIT_OCC_DIV_4K, INIT_OCC_MOD_4K;   //to get around the loop iteration problem
    INIT_OCC_DIV_4K = INITIAL_OCCUPANCY >> 12;
    INIT_OCC_MOD_4K = INITIAL_OCCUPANCY & 12'hfff;
    old_state = '0;
    for (int i=0; i<INIT_OCC_DIV_4K; i++) begin : GEN_RANDOM_BLOCK_NAME_R17
        for (int k=0; k<4096; k++) begin : GEN_RANDOM_BLOCK_NAME_R18
            for (int j=1; j<WIDTH; j++) begin : GEN_RANDOM_BLOCK_NAME_R19
                if ((j==T1) || (j==T2) || (j==T3)) new_state[j-1] = ~old_state[j] ^ old_state[0];
                else new_state[j-1] = old_state[j];
            end
            new_state[WIDTH-1] = old_state[0];
            old_state = new_state;
        end
    end
    for (int i=0; i<INIT_OCC_MOD_4K; i++) begin : GEN_RANDOM_BLOCK_NAME_R20
        for (int j=1; j<WIDTH; j++) begin : GEN_RANDOM_BLOCK_NAME_R21
            if ((j==T1) || (j==T2) || (j==T3)) new_state[j-1] = ~old_state[j] ^ old_state[0];
            else new_state[j-1] = old_state[j];
        end
        new_state[WIDTH-1] = old_state[0];
        old_state = new_state;
    end
    galois_lfsr_reset_value = old_state;
end
endfunction
localparam bit [WIDTH-1:0] RESET_VALUE = galois_lfsr_reset_value(T1, T2, T3, INITIAL_OCCUPANCY);
always_ff @(posedge clock or negedge aclrn) begin
    if (~aclrn) state <= RESET_VALUE;
    else begin
        if (enable) begin
            for (int i=1; i<WIDTH; i++) begin : GEN_RANDOM_BLOCK_NAME_R22
                if ((i == T1) || (i == T2) || (i == T3)) state[i-1] <= ~state[i] ^ state[0];
                else state[i-1] <= state[i];
            end
            state[WIDTH-1] <= state[0];
        end
        if (~sclrn) state <= RESET_VALUE;
    end
end
endmodule



module fibonacci_lfsr #(
    parameter int WIDTH,
    parameter int T1,
    parameter int INITIAL_OCCUPANCY=0
)
(
    input  wire                 clock,
    input  wire                 aclrn,
    input  wire                 sclrn,
    input  wire                 enable,
    output logic    [WIDTH-1:0] state
);
function [WIDTH-1:0] fibonacci_lfsr_reset_value;
input int T1, INITIAL_OCCUPANCY;
begin
    bit [WIDTH-1:0] old_state, new_state;
    int INIT_OCC_DIV_4K, INIT_OCC_MOD_4K;   //to get around the loop iteration problem
    INIT_OCC_DIV_4K = INITIAL_OCCUPANCY >> 12;
    INIT_OCC_MOD_4K = INITIAL_OCCUPANCY & 12'hfff;
    old_state = '0;
    for (int i=0; i<INIT_OCC_DIV_4K; i++) begin
        for (int j=0; j<4096; j++) begin
            new_state[0] = ~old_state[T1-1] ^ old_state[WIDTH-1];
            new_state[WIDTH-1:1] = old_state[WIDTH-2:0];
            old_state = new_state;
        end
    end
    for (int i=0; i<INIT_OCC_MOD_4K; i++) begin
        new_state[0] = ~old_state[T1-1] ^ old_state[WIDTH-1];
        new_state[WIDTH-1:1] = old_state[WIDTH-2:0];
        old_state = new_state;
    end
    fibonacci_lfsr_reset_value = old_state;
end
endfunction
localparam bit [WIDTH-1:0] RESET_VALUE = fibonacci_lfsr_reset_value(T1, INITIAL_OCCUPANCY);
always_ff @(posedge clock or negedge aclrn) begin
    if (~aclrn) state <= RESET_VALUE;
    else begin
        if (enable) begin
            state[0] <= ~state[T1-1] ^ state[WIDTH-1];
            state[WIDTH-1:1] <= state[WIDTH-2:0];
        end
        if (~sclrn) state <= RESET_VALUE;
    end
end
endmodule

`default_nettype wire
