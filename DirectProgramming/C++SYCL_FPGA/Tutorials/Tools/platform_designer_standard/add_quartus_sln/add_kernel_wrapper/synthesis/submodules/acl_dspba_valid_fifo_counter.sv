// (C) 1992-2012 Altera Corporation. All rights reserved.                         
// Your use of Altera Corporation's design tools, logic functions and other       
// software and tools, and its AMPP partner logic functions, and any output       
// files any of the foregoing (including device programming or simulation         
// files), and any associated documentation or information are expressly subject  
// to the terms and conditions of the Altera Program License Subscription         
// Agreement, Altera MegaCore Function License Agreement, or other applicable     
// license agreement, including, without limitation, that your use is for the     
// sole purpose of programming logic devices manufactured by Altera and sold by   
// Altera or its authorized distributors.  Please refer to the applicable         
// agreement for further details.                                                 
    


//===----------------------------------------------------------------------===//
//
// Parameterized FIFO with input and output registers and ACL pipeline
// protocol ports. This "FIFO" stores no data and only counts the number
// of valids.
//
//===----------------------------------------------------------------------===//
module acl_dspba_valid_fifo_counter 
#(
    parameter integer DEPTH = 32,           // >0
    parameter integer STRICT_DEPTH = 0,     // 0|1
    parameter integer ALLOW_FULL_WRITE = 0  // 0|1
)
(
    input logic clock,
    input logic resetn,
    input logic valid_in,
    output logic valid_out,
    input logic stall_in,
    output logic stall_out,
    output logic empty,
    output logic full
);
    // No data, so just build a counter to count the number of valids stored in this "FIFO".
    //
    // The counter is constructed to count up to a MINIMUM value of DEPTH entries.
    // * Logical range of the counter C0 is [0, DEPTH].
    // * empty = (C0 <= 0)
    // * full = (C0 >= DEPTH)
    //
    // To have efficient detection of the empty condition (C0 == 0), the range is offset
    // by -1 so that a negative number indicates empty.
    // * Logical range of the counter C1 is [-1, DEPTH-1].
    // * empty = (C1 < 0)
    // * full = (C1 >= DEPTH-1)
    // The size of counter C1 is $clog2((DEPTH-1) + 1) + 1 => $clog2(DEPTH) + 1.
    //
    // To have efficient detection of the full condition (C1 >= DEPTH-1), change the
    // full condition to C1 == 2^$clog2(DEPTH-1), which is DEPTH-1 rounded up
    // to the next power of 2. This is only done if STRICT_DEPTH == 0, otherwise
    // the full condition is comparison vs. DEPTH-1.
    // * Logical range of the counter C2 is [-1, 2^$clog2(DEPTH-1)]
    // * empty = (C2 < 0)
    // * full = (C2 == 2^$clog2(DEPTH - 1))
    // The size of counter C2 is $clog2(DEPTH-1) + 2.
    // * empty = MSB
    // * full = ~[MSB] & [MSB-1]
    localparam COUNTER_WIDTH = (STRICT_DEPTH == 0) ?
        ((DEPTH > 1 ? $clog2(DEPTH-1) : 0) + 2) :
        ($clog2(DEPTH) + 1);
    logic [COUNTER_WIDTH - 1:0] valid_counter;
    logic incr, decr;
    wire [1:0] counter_update;
    wire [COUNTER_WIDTH-1:0] counter_update_extended;

    assign counter_update = {1'b0, incr} - {1'b0, decr};
    // TODO: Remove this and replace with $signed when HSD:14021297674 is fixed
    //       and we've stopped supporting all ACDS versions that have the bug
    generate
      if (COUNTER_WIDTH-2 > 0) begin
        assign counter_update_extended = {{(COUNTER_WIDTH-2){counter_update[1]}}, counter_update};
      end else begin
        assign counter_update_extended = counter_update;
      end
    endgenerate

    assign empty = valid_counter[$bits(valid_counter) - 1];
    assign full = (STRICT_DEPTH == 0) ?
        (~valid_counter[$bits(valid_counter) - 1] & valid_counter[$bits(valid_counter) - 2]) :
        (valid_counter == DEPTH - 1);
    assign incr = valid_in & ~stall_out;
    assign decr = valid_out & ~stall_in;

    assign valid_out = ~empty;
    assign stall_out = ALLOW_FULL_WRITE ? (full & stall_in) : full;

    always @( posedge clock or negedge resetn )
        if( !resetn )
            valid_counter <= {$bits(valid_counter){1'b1}};  // -1
        else
            valid_counter <= valid_counter + counter_update_extended; 
endmodule


