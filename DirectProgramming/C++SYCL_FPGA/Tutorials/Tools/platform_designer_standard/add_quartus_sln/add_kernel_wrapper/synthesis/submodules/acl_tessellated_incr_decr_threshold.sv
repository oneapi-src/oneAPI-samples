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


// This module tracks whether FIFO occupancy is above a predefined threshold. Since the occupancy can only change by +1, 0, or -1,
// only the lower bits of the used_words counter need to be timing exact. For example, say we want to detect if occupancy >= 2. We
// activate if occupancy==1 and increasing, and we deactivate if occupancy==2 and decreasing. To detect if occupancy is 1 or 2,
// the bottom 2 bits of used_words needs to be exact, but the upper bits can be 1 clock cycle late. If the occupancy is 1 or 2 right
// now, then 1 clock cycle ago it must have been in the range of 0 to 3 inclusive. This technique is applied recursively, e.g. the
// higher bits of used_words are more stale. If we want a different threshold, then we adjust the reset value of used_words.
//
// This module was designed in conjunction with acl_high_speed_fifo.sv. The testbench is grouped together with the fifo.
//
// Required files:
// -acl_tessellated_incr_decr_threshold.sv
// -acl_reset_handler.sv
//
// TEMPORARY FIX:
// Reset values are set to match the behavior of before reset cleanup. See FB case:457213. This is a workaround for now.
// Eventually the compiler needs to set the reset parameters correctly, at which point the default values will be set
// back to the original intent, which is for someone who knows nothing about the reset in their system.

`default_nettype none

module acl_tessellated_incr_decr_threshold #(
    //general configuration
    parameter longint CAPACITY,                   // occupancy starts at INITIAL_OCCUPANCY, incr_no_overflow and decr_no_underflow should never push occupancy below 0 or above CAPACITY
    parameter longint THRESHOLD,                  // 1 or larger, up to CAPACITY, 1 checks for not empty, CAPACITY checks for full
    parameter longint INITIAL_OCCUPANCY = 0,
    
    //DEPRECATED -- case:555803 tracks the removal of this
    parameter int RESET_RELEASE_DELAY = 0,        // THIS PARAMETER HAS NO EFFECT NO MATTER WHAT VALUE YOU SET IT TO, look at the reset structure inside acl_mid_speed_fifo for how to now handle delayed reset exit
    
    //special behavior needed for some cases in fifo
    parameter bit THRESHOLD_REACHED_AT_RESET = 0, // set to 1 if you want threshold_reached to be high at reset, e.g. fifo appears full at reset
    parameter bit WRITE_AND_READ_DURING_FULL = 0, // if 1 then incr and decr can cancel when occupancy is at CAPACITY, if 0 then incr is ignored when occupancy is at CAPACITY
    parameter bit WRITE_AND_READ_DURING_EMPTY = 0,// set to 1 for use with almost_empty in zero latency fifos, behavior is different when read_used_words changes before write_used_words
    
    //reset configuration
    parameter bit ASYNC_RESET = 1,                // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 0,          // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    parameter bit RESET_EVERYTHING = 0,           // intended for partial reconfig debug, set to 1 to reset every register (normally async reset excludes data path and sync reset additionally excludes some control signals)
    parameter bit RESET_EXTERNALLY_HELD = 1       // set to 1 if resetn will be held for at least FOUR clock cycles, otherwise we will internally pulse stretch reset before consumption
)(
    input  wire     clock,
    input  wire     resetn,                       // longest reset chain is through the tessellated adder
    input  wire     incr_no_overflow,             // when tracking fifo occupancy, this is typically "wrreq & ~full"
    input  wire     incr_raw,                     // to detect when the threshold is reached, we don't care about overflow, so can use "wrreq"
    input  wire     decr_no_underflow,            // when tracking fifo occupancy, this is typically "rdreq & ~empty"
    input  wire     decr_raw,                     // to detect when the threshold is reached, we don't care about underflow, so can use "rdreq"
    output logic    threshold_reached             // asserted when occupancy >= THRESHOLD
);

    // the basic functionality of this circuit is as follows:
    // always_ff @(posedge clock) begin
    //     //decr must be off when occupancy is 0, incr must be off when occupancy == CAPACITY (unless WRITE_AND_READ_DURING_FULL = 1)
    //     occupancy <= occupancy + incr_no_overflow - decr_no_underflow;
    //     if (~resetn) occupancy <= INITIAL_OCCUPANCY;
    // end
    // assign threshold_reached = (occupancy >= THRESHOLD);
    
    // To support initial occupancy, we could instead reset the occupancy to 0 and assign threshold_reached = occupancy >= (THRESHOLD-INITIAL_OCCUPANCY).
    // This circuit was designed to detect (THRESHOLD-INITIAL_OCCUPANCY) = 0. This is the timing requirements for different bits of the occupancy counter:
    //
    // Bit range    | Name      | Max lateness  | Detection range boundary factoring in lateness
    // -------------+-----------+---------------+-------------------------------------------------
    // 1:0          | lo        | 0             | N/A - timing is exact
    // 4:2          | mid       | 1             | -2 to +1, are we in the right group of 4
    // 5+           | hi        | 13            | -18 to +13, are we in the right group of 32
    //
    // Note that mid[2] (bit 4 of occupancy) is computed at the same time as the other bits of mid, but is grouped together with the zero detect for hi. The group of 16
    // defined by mid[2] spans from -10 to +5, so actually there can be up to 5 clocks of lateness. The threshold reached happens at mid=1 instead of mid=0 so that we
    // can get some extra lateness for mid[2].
    //                                                                                                        +-------+
    // threshold - initial_occupancy: -19 -18 -17 -16 -15 -14 -13 -12 -11 -10  -9  -8  -7  -6  -5  -4  -3  -2 |-1   0 | 1   2   3   4   5   6   7   8   9  10  11  12  13  14
    //                                                                                                        +-------+
    //                            lo:   3   0   1   2   3   0   1   2   3   0   1   2   3   0   1   2   3   0 | 1   2 | 3   0   1   2   3   0   1   2   3   0   1   2   3   0
    //                                                                                                    +---+-------+---+
    //                           mid:   6   5   5   5   5   4   4   4   4   3   3   3   3   2   2   2   2 | 1   1   1   1 | 0   0   0   0   7   7   7   7   6   6   6   6   5
    //                                    +---------------------------------------------------------------+---------------+-----------------------------------------------+
    //                            hi:   1 | 0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 |-1
    //                                    +-------------------------------------------------------------------------------------------------------------------------------+
    //
    // The reset values come from the column where "threshold - initial occupancy" is 0, in this example lo=2, mid=1, and hi=0. This will detect an initial occupancy adjusted
    // threshold of 0. If we wanted to detect threshold 19 for example, the reset values simply come from 19 columns to the left, as we would need to increment 19 times
    // (moving rightwards) until we reach the threshold, in this case the reset values are lo=3, mid=6 and hi=1.
    // 
    // Notice how the group of 4 for mid is not centered inside the group of 32 for hi. While this means we cannot utilize the maximum lateness (a group size of 32 should have
    // lateness of 15), non-centered groups makes the tessellation logic simpler. This results in fewer pipeline stages that are needed to propagate a carry to the higher bits.
    // Area savings are achieved by tolerating a lower max lateness (13 is plenty enough, CAPACITY is specified on 64 bits so the occupancy cannot really use more bits).
    
    
    localparam int INIT_OCC_TO_THRESH = THRESHOLD - INITIAL_OCCUPANCY;
    localparam bit THRESHOLD_REACHED_EXIT_RESET = (THRESHOLD > INITIAL_OCCUPANCY) ? 1'b0 : 1'b1;
    
    //tessellation control - for bits 5+ of the occupancy counter, specify the minimum number of bits per tessellation stage
    //experiments indicate 4 is the best value for both fmax and area
    //a smaller value results in more pipelined registers for wrap-around detect, a value of 3 ensures everything fits within a 6-lut including the sclr
    //a larger value does not necessarily save area, quartus may decide to use a carry chain which tends to bloat the area
    //this parameter specifies the minimum size, maximum lateness is 13 so if the occupancy needs many bits then we will increase the number of bits per tessellation stage
    localparam int MIN_TESS_SIZE_HI_BITS = 4;
    
    
    
    /////////////
    //         //
    //  Reset  //
    //         //
    /////////////
    
    //count_at_target determines if the upper bits (excluding bits 1:0) are at the correct value such that a transition is allowed on threshold_reached
    //count_at_target is not reset, and neither is the zero detect on the upper bits (bit 5+)
    //the counters for occupancy are reset, but there are up to 3 pipeline stages from these to count_at_target (up to 2 for the zero detect, and one more for count_at_target)
    //therefore the reset needs to be held for 4 clock edges, 1 to reset the counters, and up to 3 more to propagate to count_at_target
    //the reason for not resetting count_at_target is to reduce the fanin logic (can fit within a 6 lut) and to reduce fanout of the reset signal (improves routability)
    
    //since we have to hold the reset for 4 clocks anyways, we can also remove the reset from the wrap-around detect registers, the pipeline stages from tessellating the counter
    //if the previous tessellation stage did not report a wrap-around on the previous clock cycle, then the current tessellation stage cannot report a wrap-around now, use this to propagate reset value 0
    //lo_3_0 has a reset, which means we can skip mid_5_6, hi.wrap[0], and hi.wrap[1] -> this will cover up to 2+3+4+4 = 13 bits of the counter
    //hi.wrap[2] has a reset, which means we can skip hi.wrap[3], hi.wrap[4], and hi.wrap[5]
    //hi.wrap[6] has a reset, which means we can skip hi.wrap[7], and hi.wrap[8]
    
    genvar g;
    logic aclrn, sclrn;
    logic sclrn_before_pulse_stretch, sclrn_pulse_stretched;
    logic [3:0] sclrn_chain;
    logic reset_exit_n;
    
    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .USE_SYNCHRONIZER       (1), // Force reset synchronization to get around the A10 reset timing issue described in HSD-ES case:14011923692
        .SYNCHRONIZE_ACLRN      (1), // Force reset synchronization to get around the A10 reset timing issue described in HSD-ES case:14011923692
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
    if (ASYNC_RESET) begin : gen_no_sclrn
        assign sclrn = 1'b1;
    end
    else if (RESET_EXTERNALLY_HELD) begin : gen_direct_sclrn    //this is the typical usage from within hld_fifo on s10
        assign sclrn = sclrn_before_pulse_stretch;
    end
    else begin : gen_pulse_stretch_sclrn
        always_ff @(posedge clock) begin
            sclrn_chain <= (sclrn_chain << 1) | sclrn_before_pulse_stretch;
            sclrn_pulse_stretched <= &sclrn_chain;
            sclrn <= sclrn_pulse_stretched;
        end
    end
    endgenerate
    
    //assuming occupancy starts at 0, so if threshold_reached needs to be asserted during reset, we need a way to deassert it after reset is released
    //the only thing consumed from this block below is the reset_exit_n signal
    generate
    if (THRESHOLD_REACHED_EXIT_RESET == THRESHOLD_REACHED_AT_RESET) begin   //don't need to flip value of threshold_reached at end of reset
        assign reset_exit_n = 1'b1; //this is active low
    end
    else begin
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) reset_exit_n <= 1'b0;
            else begin
                reset_exit_n <= sclrn;
            end
        end
    end
    endgenerate
    
    
    //////////////////////////
    //                      //
    //  Occupancy tracking  //
    //                      //
    //////////////////////////
    
    generate
    if (THRESHOLD < 1) begin : THRESHOLD_REACHED_ONE
        assign threshold_reached = 1'b1;
    end
    else if (THRESHOLD > CAPACITY) begin : THRESHOLD_REACHED_ZERO
        assign threshold_reached = 1'b0;
    end
    else if (CAPACITY == 64'd1) begin : CAPACITY_ONE
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) threshold_reached <= THRESHOLD_REACHED_AT_RESET;
            else begin
                if (incr_no_overflow & ((WRITE_AND_READ_DURING_EMPTY) ? ~decr_no_underflow : 1'b1)) threshold_reached <= 1'b1;
                if (decr_no_underflow & ((WRITE_AND_READ_DURING_FULL) ? ~incr_no_overflow : 1'b1)) threshold_reached <= 1'b0;
                if (~reset_exit_n) threshold_reached <= THRESHOLD_REACHED_EXIT_RESET;
                if (~sclrn) threshold_reached <= THRESHOLD_REACHED_AT_RESET;
            end
        end
    end
    else begin : INCR_DECR
        ////////////////////////////////////////////////////////////////////////////////////////////
        //                                                                                        //
        //  Tessellated counters to track the occupancy, upper bits are progressively more stale  //
        //                                                                                        //
        ////////////////////////////////////////////////////////////////////////////////////////////
        
        localparam bit [63:0] TWO_MINUS_INIT_OCC_TO_THRESH = 64'd2 - INIT_OCC_TO_THRESH;
        localparam bit [1:0] LO_RESET = TWO_MINUS_INIT_OCC_TO_THRESH[1:0];
        localparam bit [63:0] FIVE_PLUS_INIT_OCC_TO_THRESH = 64'd5 + INIT_OCC_TO_THRESH;
        localparam bit [2:0] MID_RESET = FIVE_PLUS_INIT_OCC_TO_THRESH[4:2];
        
        logic [1:0] lo;                 //bits [1:0] of occupancy
        logic [2:0] mid;                //bits [4:2] of occupancy, reverse order
        logic       lo_3_0;             //asserts when lo transitions between 3 and 0 -> like an enable for mid
        logic       mid_5_6;            //asserts when mid transitions between 3 and 4 -> like an enable for hi
        logic       count_at_target;    //are bits [WIDTH-1:2] at the correct value such that a transition on threshold_reached may happen, represents the collection of mid and hi
        logic       upper_zeros;        //like count_at_target, but applicable only to bits [WIDTH-1:5]
        
        //bits 1:0, exact timing
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                lo <= LO_RESET;
                lo_3_0 <= 1'b0;
            end
            else begin
                //a more quartus-friedly version of:
                //lo <= lo + incr_no_overflow - decr_no_underflow
                lo[0] <= lo[0] ^ incr_no_overflow ^ decr_no_underflow;
                lo[1] <= lo[1] ^ (incr_no_overflow & ~decr_no_underflow & lo[0]) ^ (~incr_no_overflow & decr_no_underflow & ~lo[0]);
                
                //assert when lo transitions between 3 and 0, use next state logic (3 and incrementing, or 0 and decrementing) so that registered signal asserts when the wrap actually happens
                lo_3_0 <= (incr_no_overflow & ~decr_no_underflow & (lo==2'h3)) | (~incr_no_overflow & decr_no_underflow & (lo==2'h0));
                if (~sclrn) begin
                    lo <= LO_RESET;
                    lo_3_0 <= 1'b0;
                end
            end
        end
        
        //bits 4:2 reverse order, 1 clock late
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                mid <= MID_RESET;
                mid_5_6 <= 1'b0;
            end
            else begin
                if (lo_3_0) begin                   //lo has wrapped around between 0 and 3
                    if (lo[0]) mid <= mid + 1'b1;   //lo is currently odd, therefore it must have went from 0 to 3 -> although lo decremented, mid increments, this is where the reversal happens
                    else mid <= mid - 1'b1;
                end
                
                //assert when mid transitions between 5 and 6, use next state logic (5 and incrementing, or 6 and decrementing) so that registered signal asserts when the wrap actually happens
                mid_5_6 <= lo_3_0 & ((lo[0]) ? (mid==3'h5) : (mid==3'h6));
                if (~sclrn) begin
                    mid <= MID_RESET;
                    if (RESET_EVERYTHING) mid_5_6 <= 1'b0;
                end
            end
        end
        
        //bits 5+ reverse order, 2+ clocks late, uppermost bits can be up to 10 clocks late, zero detect can be up to 12 clocks late
        //the only signal consumed from this block is upper_zeros, which is a stale version of checking if {hi,mid[2]} == 0
        if (CAPACITY <= 16) begin
            assign upper_zeros = 1'b1;
        end
        else if (CAPACITY <= 32) begin
            assign upper_zeros = ~mid[2];
        end
        else begin : HIGH_CAPACITY
            localparam int HI_BITS = $clog2(CAPACITY) - 5;
            localparam bit [63:0] THIRTEEN_PLUS_INIT_OCC_TO_THRESH = 64'd13 + INIT_OCC_TO_THRESH;
            localparam bit [HI_BITS-1:0] HI_RESET = THIRTEEN_PLUS_INIT_OCC_TO_THRESH[HI_BITS+4:5];
            
            localparam int TESS_STAGES_RAW = (HI_BITS+MIN_TESS_SIZE_HI_BITS-1) / MIN_TESS_SIZE_HI_BITS;     //ceiling(hi_bits/tess_size)
            localparam int TESS_STAGES = (TESS_STAGES_RAW >= 9) ? 9 : TESS_STAGES_RAW;                      //clip at 9 stages, stage 0 is 2 clocks late, stage 8 is 10 clocks late
            
            logic [HI_BITS-1:0]   hi;       //the actual bits of the tessellated counter
            logic [TESS_STAGES:0] wrap;     //index n+1 indicates stage n has wrapped-around - asserts on the same clock cycle as the counter actually wrapping around
            logic [TESS_STAGES:0] odd;      //this is just the lsb of the counter from each tessellation stage
            
            assign wrap[0] = mid_5_6;       //previous values for first tessellation stage come from the mid counter
            assign odd[0] = mid[0];
            
            for (g=0; g<TESS_STAGES; g++) begin : HI_TESS
                localparam int START = g * HI_BITS / TESS_STAGES;       //inclusive
                localparam int END = (g+1) * HI_BITS / TESS_STAGES;     //exclusive
                
                always_ff @(posedge clock or negedge aclrn) begin
                    if (~aclrn) begin
                        hi[END-1:START] <= HI_RESET[END-1:START];
                        wrap[g+1] <= 1'b0;
                    end
                    else begin
                        if (wrap[g]) begin
                            if (odd[g]) hi[END-1:START] <= hi[END-1:START] - 1'b1;
                            else hi[END-1:START] <= hi[END-1:START] + 1'b1;
                        end
                        //current stage wraps arounds when (0 and decrementing) | (MAX_VALUE and incrementing), use next state logic so that registered signal asserts when the wrap actually happens
                        wrap[g+1] <= (wrap[g] & odd[g] & ~(|hi[END-1:START])) | (wrap[g] & ~odd[g] & (&hi[END-1:START]));
                        if (~sclrn) begin
                            hi[END-1:START] <= HI_RESET[END-1:START];
                            if (g==2 || g==6) wrap[g+1] <= 1'b0;
                            else if (RESET_EVERYTHING) wrap[g+1] <= 1'b0;
                        end
                    end
                end
                assign odd[g+1] = hi[START];
            end
            
            //hi can be up to 10 clocks late, use up to 2 clocks to determine if all bits are 0
            //upper_zeros is up to 12 clocks late
            localparam int HI_AND_MID2_BITS = HI_BITS + 1;
            logic [HI_AND_MID2_BITS-1:0] hi_and_mid2;
            assign hi_and_mid2 = {hi, mid[2]};
            
            if (HI_AND_MID2_BITS <= 6) begin : UPPER_ZEROS_ONE
                always_ff @(posedge clock) begin
                    upper_zeros <= (hi_and_mid2 == 0);
                end
            end
            else if (HI_AND_MID2_BITS <= 11) begin : UPPER_ZEROS_ONE_PLUS   //instead of grouping 6 and 5 inputs and then merging the result, we can fit the 5 within the second stage merge
                logic group;
                always_ff @(posedge clock) begin
                    group <= (hi_and_mid2[5:0] == 0);
                    upper_zeros <= group & (hi_and_mid2[HI_AND_MID2_BITS-1:6] == 0);
                end
            end
            //we could extend the trick used in UPPER_ZEROS_ONE_PLUS to save one register, but it is not common for fifos to need so much capacity
            else begin : UPPER_ZEROS_TWO
                //first register stage collects groups of 6, second register stage merges all together
                localparam int GROUPS = (HI_AND_MID2_BITS+5) / 6;   //ceiling(hi_and_mid2_bits/6)
                logic [GROUPS-1:0] group;
                always_ff @(posedge clock) begin
                    for (int i=0; i<GROUPS-1; i++) group[i] <= (hi_and_mid2[6*i+:6] == 0);
                    group[GROUPS-1] <= (hi_and_mid2[HI_AND_MID2_BITS-1:6*(GROUPS-1)] == 0);
                    upper_zeros <= &group;
                end
            end
        end
        
        
        
        //////////////////////////////////////////////////////////
        //                                                      //
        //  Can a transition even happen on threshold_reached?  //
        //                                                      //
        //////////////////////////////////////////////////////////
        
        //determine if the upper bits (exluding bits 1:0) are at the value at which a transition is allowed for threshold_reached
        //let quartus trim away stuff that isn't needed at smaller bit width
        if (CAPACITY <= 4) begin
            assign count_at_target = 1'b1;
        end
        else if (CAPACITY <= 8) begin
            assign count_at_target = mid[0];
        end
        else begin
            always_ff @(posedge clock) begin
                //mid[1:0] next state 1: (1 and no change) or (0 and incrementing) or (2 and decrementing)
                count_at_target <= ((~lo_3_0) ? (mid[1:0]==2'h1) : (lo[0]) ? (mid[1:0]==2'h0) : (mid[1:0]==2'h2)) & upper_zeros;
            end
        end
        
        
        
        //////////////////////////////////////////////
        //                                          //
        //  Check if we are crossing the threshold  //
        //                                          //
        //////////////////////////////////////////////
        
        if (THRESHOLD == 1 && !WRITE_AND_READ_DURING_EMPTY) begin               //checking for not empty
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) threshold_reached <= THRESHOLD_REACHED_AT_RESET;
                else begin
                    if (decr_raw & (lo==2'h2) & count_at_target) threshold_reached <= 1'b0;
                    if (incr_raw) threshold_reached <= 1'b1;
                    if (~reset_exit_n) threshold_reached <= THRESHOLD_REACHED_EXIT_RESET;
                    if (~sclrn) threshold_reached <= THRESHOLD_REACHED_AT_RESET;
                end
            end
        end
        else if (THRESHOLD == CAPACITY && !WRITE_AND_READ_DURING_FULL) begin    //checking for full
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) threshold_reached <= THRESHOLD_REACHED_AT_RESET;
                else begin
                    if (incr_raw & (lo==2'h1) & count_at_target) threshold_reached <= 1'b1;
                    if (decr_raw) threshold_reached <= 1'b0;
                    if (~reset_exit_n) threshold_reached <= THRESHOLD_REACHED_EXIT_RESET;
                    if (~sclrn) threshold_reached <= THRESHOLD_REACHED_AT_RESET;
                end
            end
        end
        else begin                              //if threshold is not near full or empty, don't need to guard against overflow/underflow
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) threshold_reached <= THRESHOLD_REACHED_AT_RESET;
                else begin
                    if (incr_raw & ~decr_raw & (lo==2'h1) & count_at_target) threshold_reached <= 1'b1;
                    if (~incr_raw & decr_raw & (lo==2'h2) & count_at_target) threshold_reached <= 1'b0;
                    if (~reset_exit_n) threshold_reached <= THRESHOLD_REACHED_EXIT_RESET;
                    if (~sclrn) threshold_reached <= THRESHOLD_REACHED_AT_RESET;
                end
            end
        end
    end
    endgenerate
    
endmodule

`default_nettype wire
