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


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                          //
//  ACL ZERO LATENCY FIFO                                                                                                                   //
//  Designed and optimized by: Jason Thong                                                                                                  //
//                                                                                                                                          //
//  DESCRIPTION                                                                                                                             //
//  ===========                                                                                                                             //
//  This fifo has a write to read latency of zero clock cycles. On the same clock cycle that data is written, it is also readable. This is  //
//  achieved by adding a data bypass using combinational logic. This is built on top of acl_low_latency_fifo, which is a fifo with write    //
//  to read latency of 1. The zero latency fifo has a very large fmax penalty due to the combinational logic between inputs and outputs.    //
//                                                                                                                                          //
//  REQUIRED FILES                                                                                                                          //
//  ==============                                                                                                                          //
//  - acl_zero_latency_fifo.sv                                                                                                              //
//  - acl_low_latency_fifo.sv                                                                                                               //
//  - acl_reset_handler.sv                                                                                                                  //
//  - acl_fanout_pipeline.sv                                                                                                                //
//  - acl_std_synchronizer_nocut.sv                                                                                                         //
//                                                                                                                                          //
//  INTERPRETATION OF ZERO LATENCY                                                                                                          //
//  ==============================                                                                                                          //
//  Zero write to read latency can be a tricky concept. Typically in a fifo there are two occupancy trackers:                               //
//                                                                                                                                          //
//  1. write_used_words:                                                                                                                    //
//       - tracks how many words have been written into the fifo                                                                            //
//       - increments one clock after write (on the next clock edge when write is asserted), decrements one clock after read ack            //
//       - stall_out (fifo is full) and almost_full are based on this value                                                                 //
//  2. read_used_words:                                                                                                                     //
//       - tracks how many words are readable from the fifo                                                                                 //
//       - increments WRITE_TO_READ_LATENCY clocks after write, decrements one clock after read ack                                         //
//       - valid_out (fifo is not empty) and almost_empty are based on this value                                                           //
//                                                                                                                                          //
//  In a high latency fifo e.g. acl_high_speed_fifo with WRITE_TO_READ_LATENCY = 5, write_used_words increments before read_used_words.     //
//  The interpretation is that something written into the fifo takes some time before that item is readable. In acl_low_latency_fifo        //
//  we have WRITE_TO_READ_LATENCY = 1, so write_used_words has the same timing as read_used_words. In acl_zero_latency_fifo, notice that    //
//  read_used_words increments BEFORE write_used_words. It may sound like something is readable before it has been written to the fifo,     //
//  but the interpretation here is that data can bypass the storage associated with a fifo e.g. data does not need to be written to a       //
//  register or a memory in order to be readable.                                                                                           //
//                                                                                                                                          //
//  RESET BEHAVIOR                                                                                                                          //
//  ==============                                                                                                                          //
//  The fifo asserts full (stall_out) and empty (~valid_out) during reset. Refer to acl_high_speed_fifo.sv for full details.                //
//                                                                                                                                          //
//  ALMOST FULL AND ALMOST EMPTY                                                                                                            //
//  ============================                                                                                                            //
//  The ALMOST_***_CUTOFF parameters refer to how much dead space would be in the fifo if one were to use almost_full as same clock cycle   //
//  backpressure (dead space in not being able to completely fill the fifo), or if one were to almost_empty as same clock cycle underflow   //
//  prevention (dead space in not being able to empty the fifo). See chart below for interpretation of values:                              //
//                                                                                                                                          //
//  Scfifo parameter                    | Our parameter             | Interpretation                                                        //
//  ------------------------------------+---------------------------+---------------------------------------------------------------        //
//  almost_empty_value = 1              | ALMOST_EMPTY_CUTOFF = 0   | almost_empty behaves the same way as empty                            //
//  almost_empty_value = 2              | ALMOST_EMPTY_CUTOFF = 1   | almost_empty asserts when read_used_words is 1 or less                //
//  ------------------------------------+---------------------------+---------------------------------------------------------------        //
//  almost_full_value = lpm_numwords    | ALMOST_FULL_CUTOFF = 0    | almost_full behaves the same way as full                              //
//  almost_full_value = lpm_numwords-1  | ALMOST_FULL_CUTOFF = 1    | almost_full asserts when write_used_words is DEPTH-1 or higher        //
//                                                                                                                                          //
//  INITIAL OCCUPANCY                                                                                                                       //
//  =================                                                                                                                       //
//  The parameter INITIAL_OCCUPANCY describes the number of words of garbage data in the fifo as it exits from reset. Typically this is 0,  //
//  e.g. we have to write into the fifo  before anything is readable. If INITIAL_OCCUPANCY > 0, then valid_out is 0 during reset, and when  //
//  it eventually asserts it is then safe for downstream to transact reads from the fifo. Exit from reset should be handled separately for  //
//  upstream and downstream. In particular, the assertion of valid_out (to downstream) and the deassertion of stall_out (to upstream) may   //
//  not happen on the same clock cycle. If INITIAL_OCCUPANCY == DEPTH, one cannot use stall_out to observe reset exit, only when at least   //
//  one item has been read from the fifo will stall_out then deasert.                                                                       //
//                                                                                                                                          //
//  VALID_IN_EARLINESS AND STALL_IN_EARLINESS                                                                                               //
//  ============================                                                                                                            //
//  This is an fmax optimization. If valid_in and stall_in are known one clock cycle ahead of time, we compute occupancy one clock early.   //
//  This gives an extra clock cycle from the control logic (occupancy, clock enables, mux selects) to the data path logic, which helps      //
//  mitigate fmax degradation due to high fan-out (large WIDTH). Only values 0 and 1 are supported since the biggest fmax gain comes from   //
//  registering the control logic before consumption. Any excess earliness is absorbed by hld_fifo, we leave it to Quartus to retime it.    //
//                                                                                                                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TESTBENCH: the testbench for this has now been merged into testbench of hld_fifo


`default_nettype none

module acl_zero_latency_fifo #(
    //parameters are mostly passed directly into acl_low_latency_fifo, see that module for detailed descriptions, we will highlight the important differences here
    //basic fifo configuration
    parameter int WIDTH,
    parameter int DEPTH,
    
    //occupancy
    parameter int ALMOST_EMPTY_CUTOFF = 0,      // almost_empty asserts if read_used_words <= ALMOST_EMPTY_CUTOFF, note that read_used_words changes before write_used_words at zero write to read latency
    parameter int ALMOST_FULL_CUTOFF = 0,
    parameter int INITIAL_OCCUPANCY = 0,
    
    //reset configuration
    parameter bit ASYNC_RESET = 1,
    parameter bit SYNCHRONIZE_RESET = 0,
    parameter bit RESET_EVERYTHING = 0,
    parameter bit RESET_EXTERNALLY_HELD = 0,
    
    //special configurations for higher fmax / lower area
    parameter int STALL_IN_EARLINESS = 0,
    parameter int VALID_IN_EARLINESS = 0,
    parameter int STALL_IN_OUTSIDE_REGS = 0,    // number of registers on the stall-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int VALID_IN_OUTSIDE_REGS = 0,    // number of registers on the valid-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int REGISTERED_DATA_OUT_COUNT = 0,// this affects fifo_data_out (the data_out inside acl_low_latency_fifo), not the output of acl_zero_latency_fifo which can never be registered due to the bypass
    parameter bit NEVER_OVERFLOWS = 0,
    
    //special features that typically have an fmax penalty
    parameter bit HOLD_DATA_OUT_WHEN_EMPTY = 0,
    parameter bit WRITE_AND_READ_DURING_FULL = 0
)
(
    input  wire                 clock,
    input  wire                 resetn,         // see description above for different reset modes
    
    //write interface
    input  wire                 valid_in,       // upstream advertises it has data, a write happens when valid_in & ~stall_out -- this needs to be early if VALID_IN_EARLINESS >= 1
    input  wire     [WIDTH-1:0] data_in,        // data from upstream
    output logic                stall_out,      // inform upstream that we cannot accept data
    output logic                almost_full,    // asserts if write_used_words >= (DEPTH-ALMOST_FULL_CUTOFF)
    
    //read interface
    output logic                valid_out,      // advertise to downstream that we have data
    output logic    [WIDTH-1:0] data_out,       // data to downstream
    input  wire                 stall_in,       // downstream indicates it cannot accept data -- this needs to be early if STALL_IN_EARLINESS >= 1
    output logic                almost_empty,   // asserts if read_used_words <= ALMOST_EMPTY_CUTOFF
    output logic                forced_read_out // indicates fifo is being read on current clock cycle, read data must be consumed or it will be lost, is a registered signal if STALL_IN_EARLINESS >= 1 && VALID_IN_EARLINESS >= 1
);
    
    //////////////////////////////////////
    //                                  //
    //  Sanity check on the parameters  //
    //                                  //
    //////////////////////////////////////
    
    // do not allow arbitrarily large amounts of earliness, as this delays the exit from reset "safe state"
    // the checks are done in Quartus pro and Modelsim, it is disabled in Quartus standard because it results in a syntax error (parser is based on an older systemverilog standard)
    // the workaround is to use synthesis translate to hide this from Quartus standard, ALTERA_RESERVED_QHD is only defined in Quartus pro, and Modelsim ignores the synthesis comment
    `ifdef ALTERA_RESERVED_QHD
    `else
    //synthesis translate_off
    `endif
    generate
    if (DEPTH < 1) begin
        $fatal(1, "acl_zero_latency_fifo: illegal value of DEPTH = %d, minimum allowed is 1\n", DEPTH);
    end
    if ((ALMOST_EMPTY_CUTOFF < 0) || (ALMOST_EMPTY_CUTOFF > DEPTH)) begin
        $fatal(1, "acl_zero_latency_fifo: illegal value of ALMOST_EMPTY_CUTOFF = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", ALMOST_EMPTY_CUTOFF, DEPTH);
    end
    if ((ALMOST_FULL_CUTOFF < 0) || (ALMOST_FULL_CUTOFF > DEPTH)) begin
        $fatal(1, "acl_zero_latency_fifo: illegal value of ALMOST_FULL_CUTOFF = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", ALMOST_FULL_CUTOFF, DEPTH);
    end
    if ((INITIAL_OCCUPANCY < 0) || (INITIAL_OCCUPANCY > DEPTH)) begin
        $fatal(1, "acl_zero_latency_fifo: illegal value of INITIAL_OCCUPANCY = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", INITIAL_OCCUPANCY, DEPTH);
    end
    if ((REGISTERED_DATA_OUT_COUNT < 0) || (REGISTERED_DATA_OUT_COUNT > WIDTH)) begin
        $fatal(1, "acl_zero_latency_fifo: illegal value of REGISTERED_DATA_OUT_COUNT = %d, minimum allowed is 0, maximum allowed is WIDTH = %d\n", REGISTERED_DATA_OUT_COUNT, WIDTH);
    end
    if ((STALL_IN_EARLINESS < 0) || (STALL_IN_EARLINESS > 10)) begin
        $fatal(1, "acl_zero_latency_fifo: illegal value of STALL_IN_EARLINESS = %d, minimum allowed is 0, maximum allowed is 10\n", STALL_IN_EARLINESS);
    end
    if ((VALID_IN_EARLINESS < 0) || (VALID_IN_EARLINESS > 10)) begin
        $fatal(1, "acl_zero_latency_fifo: illegal value of VALID_IN_EARLINESS = %d, minimum allowed is 0, maximum allowed is 10\n", VALID_IN_EARLINESS);
    end
    if (VALID_IN_OUTSIDE_REGS < 0 || VALID_IN_OUTSIDE_REGS > 1) begin
        $fatal(1, "acl_high_speed_fifo: illegal value of VALID_IN_OUTSIDE_REGS = %d, minimum allowed is 0, maximum allowed is 1\n", VALID_IN_OUTSIDE_REGS);
    end
    if (STALL_IN_OUTSIDE_REGS < 0 || STALL_IN_OUTSIDE_REGS > 1) begin
        $fatal(1, "acl_high_speed_fifo: illegal value of STALL_IN_OUTSIDE_REGS = %d, minimum allowed is 0, maximum allowed is 1\n", STALL_IN_OUTSIDE_REGS);
    end
    endgenerate
    `ifdef ALTERA_RESERVED_QHD
    `else
    //synthesis translate_on
    `endif
    
    
    
    //////////////////////////
    //                      //
    //  Parameter settings  //
    //                      //
    //////////////////////////
    
    //excess earliness
    localparam int EARLINESS                = ((STALL_IN_EARLINESS >= 1) && (VALID_IN_EARLINESS >= 1)) ? 1 : 0; //how much earliness on stall_in and valid_in will the fifo take advantage of
    localparam int EXCESS_EARLY_STALL       = STALL_IN_EARLINESS - EARLINESS;                                   //how many pipeline stages to use to absorb excess STALL_IN_EARLINESS
    localparam int EXCESS_EARLY_VALID       = VALID_IN_EARLINESS - EARLINESS;                                   //how many pipeline stages to use to absorb excess VALID_IN_EARLINESS
    
    //reset timing
    localparam int EXCESS_EARLY_STALL_WITH_EXT = STALL_IN_EARLINESS + STALL_IN_OUTSIDE_REGS;    //early stall is affected by regs outside this module; account for effect on reset timing
    localparam int EXCESS_EARLY_VALID_WITH_EXT = VALID_IN_EARLINESS + VALID_IN_OUTSIDE_REGS;    //early valid is affected by regs outisde this module; account for effect on reset timing
    localparam int FLUSH_EARLY_PIPES        = (EXCESS_EARLY_STALL_WITH_EXT > EXCESS_EARLY_VALID_WITH_EXT) ? EXCESS_EARLY_STALL_WITH_EXT : EXCESS_EARLY_VALID_WITH_EXT;  //clocks needs to flush excess earliness pipelines
    localparam int RESET_SYNC_DEPTH         = (SYNCHRONIZE_RESET) ? 3 : 0;                                      //how many registers are added inside acl_reset_handler for synchronizing the reset
    localparam int RESET_PIPE_DEPTH         = 2;                                                                //how many pipeline stages we add to sclrn
    localparam int RESET_LATENCY            = (ASYNC_RESET || RESET_EVERYTHING) ? 0 : (RESET_SYNC_DEPTH + RESET_PIPE_DEPTH);    //how many clocks from the resetn input signal until the reset is consumed
    localparam int MIN_RESET_RELEASE_DELAY  = EARLINESS;                                                        //when earliness = 1 occ is retimed one clock earlier, need to delay fifo exit from safe state by at least 1 clock
    localparam int RAW_RESET_RELEASE_DELAY  = FLUSH_EARLY_PIPES - RESET_LATENCY;                                //delay fifo exit from safe state if need more clocks to flush earliness than reset latency
    localparam int RESET_RELEASE_DELAY      = (RAW_RESET_RELEASE_DELAY < MIN_RESET_RELEASE_DELAY) ? MIN_RESET_RELEASE_DELAY : RAW_RESET_RELEASE_DELAY;  //how many clocks late the fifo exits from safe state
    
    // properties of the fifo which are consumed by the testbench
    localparam int WRITE_TO_READ_LATENCY            = 0;    //once something is written into the fifo, how many clocks later will it be visible on the read side
    localparam int RESET_EXT_HELD_LENGTH            = 1;    //how many clocks does reset need to be held for, this fifo does not take advantage of RESET_EXTERNALLY_HELD
    localparam int MAX_CLOCKS_TO_ENTER_SAFE_STATE   = 5;    //upon assertion of reset, worse case number of clocks until fifo shows both full and empty
    localparam int MAX_CLOCKS_TO_EXIT_SAFE_STATE    = 15;   //upon release of reset, worse case number of clocks until fifo is ready to transact (not necessarily observable if INITIAL_OCCUPANCY = DEPTH)
    
    
    
    ///////////////////////////
    //                       //
    //  Signal declarations  //
    //                       //
    ///////////////////////////
    
    //reset
    genvar            g;
    logic             aclrn, sclrn_pre, sclrn, sclrn_reset_everything, resetn_synchronized;
    logic [RESET_RELEASE_DELAY:0] resetn_delayed;
    logic             fifo_in_reset, occ_in_reset;
    
    //retime stall_in and valid_in to the correct timing, absorb excess earliness that the fifo cannot take advantage of
    logic stall_in_correct_timing, valid_in_correct_timing;
    logic [EXCESS_EARLY_STALL:0] stall_in_pipe;
    logic [EXCESS_EARLY_VALID:0] valid_in_pipe;
    
    //signals for wrapping around low latency fifo to get zero latency behavior
    logic fifo_valid_in, fifo_valid_out;
    logic [WIDTH-1:0] fifo_data_out;
    logic [DEPTH-1:0] occ_low_reset, occ_high_reset;
    logic almost_empty_if_not_writing, almost_empty_if_writing, valid_in_correct_timing_mask;
    logic almost_empty_pre, valid_out_pre;
    
    
    
    /////////////
    //         //
    //  Reset  //
    //         //
    /////////////
    
    // IMPORTANT: reset behavior must EXACTLY match acl_low_latency_fifo
    
    // S10 reset specification:
    // S (clocks to enter reset "safe state"): 2 (sclrn pipeline, one inside acl_reset_handler, one which we manually create)
    // P (minimum duration of reset pulse):    1 (no pulse stretcher is needed)
    // D (clocks to exit reset "safe state"):  15 (3 for synchronizer) + (2 for sclrn pipeline) + (10 for reset release delay for registers that absorb excess earliness)
    
    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
        .PIPE_DEPTH             (1),    //we add one more pipeline stage ourselves, needed so that resetn_delayed can use sclrn_pre
        .NUM_COPIES             (1)
    )
    acl_reset_handler_inst
    (
        .clk                    (clock),
        .i_resetn               (resetn),
        .o_aclrn                (aclrn),
        .o_resetn_synchronized  (resetn_synchronized),
        .o_sclrn                (sclrn_pre)
    );
    always_ff @(posedge clock) begin
        sclrn <= sclrn_pre;
    end
    assign sclrn_reset_everything = (RESET_EVERYTHING) ? sclrn : 1'b1;
    
    generate
    always_comb begin
        resetn_delayed[0] = (ASYNC_RESET) ? aclrn : sclrn;
    end
    for (g=1; g<=RESET_RELEASE_DELAY; g++) begin : gen_resetn_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) resetn_delayed[g] <= '0;
            else begin
                resetn_delayed[g] <= (ASYNC_RESET && g==1) ? 1'b1 : resetn_delayed[g-1];    //partial reconfig requires no d-input consumption of aclr, so resetn_delayed[1] loads 1'b1 if ASYNC_RESET
                if (~sclrn_pre) resetn_delayed[g] <= '0;    //resetn_delayed goes into reset as the same time as sclrn, since this is registered need to peek one clock ahead of sclrn
            end
        end
    end
    endgenerate
    
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            fifo_in_reset <= 1'b1;
            occ_in_reset <= 1'b1;
        end
        else begin
            fifo_in_reset <= 1'b0;
            occ_in_reset <= 1'b0;
            if (~resetn_delayed[RESET_RELEASE_DELAY]) fifo_in_reset <= 1'b1;
            if (~resetn_delayed[RESET_RELEASE_DELAY-EARLINESS]) occ_in_reset <= 1'b1;
        end
    end
    
    
    
    ////////////////////////////////////////////////
    //                                            //
    //  Absorb excess earliness on input signals  //
    //                                            //
    ////////////////////////////////////////////////
    
    // the fifo cannot take adnvatage of STALL_IN_EARLINESS above 1 and VALID_IN_EARLINESS above 1
    // to take advantage of earliness, both STALL_IN_EARLINESS and VALID_IN_EARLINESS need to be 1
    // absorb the excess earliness with registers, and provide the correctly timed version based on the amount of earliness that the fifo will use
    generate
    always_comb begin
        stall_in_pipe[0] = stall_in;
        valid_in_pipe[0] = valid_in;
    end
    for (g=1; g<=EXCESS_EARLY_STALL; g++) begin : gen_stall_in_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) stall_in_pipe[g] <= (RESET_EVERYTHING) ? 1'b1 : 1'bx;
            else begin
                stall_in_pipe[g] <= stall_in_pipe[g-1];
                if (~sclrn_reset_everything) stall_in_pipe[g] <= 1'b1;
            end
        end
    end
    for (g=1; g<=EXCESS_EARLY_VALID; g++) begin : gen_valid_in_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) valid_in_pipe[g] <= (RESET_EVERYTHING) ? 1'b0 : 1'bx;
            else begin
                valid_in_pipe[g] <= valid_in_pipe[g-1];
                if (~sclrn_reset_everything) valid_in_pipe[g] <= 1'b0;
            end
        end
    end
    endgenerate
    
    //these signals either both have EARLINESS == 0 or both have EARLINESS == 1
    assign stall_in_correct_timing = stall_in_pipe[EXCESS_EARLY_STALL];
    assign valid_in_correct_timing = valid_in_pipe[EXCESS_EARLY_VALID];
    
    
    
    //occ_***_reset and the ****_pre signals are retimed early like valid_in and stall_in when EARLINESS == 1
    
    acl_low_latency_fifo
    #(
        .WIDTH                          (WIDTH),
        .DEPTH                          (DEPTH),
        .ALMOST_EMPTY_CUTOFF            (0),    //almost_empty is managed here because read_used_words changes before occ
        .ALMOST_FULL_CUTOFF             (ALMOST_FULL_CUTOFF),
        .INITIAL_OCCUPANCY              (INITIAL_OCCUPANCY),
        .ASYNC_RESET                    (ASYNC_RESET),
        .SYNCHRONIZE_RESET              (0),    //do not synchronize the reset again, this ensures that e.g. stall_out deasserting upon reset exit happens on the same clock cycle here and well as inside low latency fifo
        .RESET_EVERYTHING               (RESET_EVERYTHING),
        .RESET_EXTERNALLY_HELD          (RESET_EXTERNALLY_HELD),
        .STALL_IN_EARLINESS             (EARLINESS),    //we have to absorb excess earliness so that valid_in and stall_in have the same earliness (needed for bypass logic), and that earliness can only be 0 or 1...
        .VALID_IN_EARLINESS             (EARLINESS),
        .REGISTERED_DATA_OUT_COUNT      (REGISTERED_DATA_OUT_COUNT),
        .NEVER_OVERFLOWS                (NEVER_OVERFLOWS),
        .HOLD_DATA_OUT_WHEN_EMPTY       (0),
        .WRITE_AND_READ_DURING_FULL     (WRITE_AND_READ_DURING_FULL),
        .RESET_RELEASE_DELAY_OVERRIDE_FROM_ZERO_LATENCY_FIFO(RESET_RELEASE_DELAY)   //...but if valid_in or stall_in are very early, we still need the fifo to exit from reset safe state later than usual
    )
    ll_fifo_inst
    (
        .clock                          (clock),
        .resetn                         (resetn_synchronized),
        
        .valid_in                       (fifo_valid_in),
        .data_in                        (data_in),
        .stall_out                      (stall_out),
        .almost_full                    (almost_full),
        
        .valid_out                      (fifo_valid_out),
        .data_out                       (fifo_data_out),
        .stall_in                       (stall_in_correct_timing),
        .almost_empty                   (),             //we have to manage this since read_used_words updates sooner than write_used_words
        .forced_read_out                (),
        
        .occ                            (),
        .occ_low_reset                  (occ_low_reset),
        .occ_high_reset                 (occ_high_reset)
    );
    
    //write into the fifo if the zero latency bypass is not used
    assign fifo_valid_in = valid_in_correct_timing & (stall_in_correct_timing | occ_low_reset[0]);
    
    //occ tracks write_used_words, convert to read_used_words by incrementing if writing to fifo (write can bypass storage)
    //stall_out gates valid_in when the fifo is full and also during reset (during which the fifo advertises that it is full)
    generate
    if (ALMOST_EMPTY_CUTOFF == DEPTH) begin
        assign almost_empty_if_not_writing = 1'b1;
    end
    else begin
        assign almost_empty_if_not_writing = ~occ_low_reset[ALMOST_EMPTY_CUTOFF];
    end
    endgenerate
    generate
    if (ALMOST_EMPTY_CUTOFF == 0) begin
        assign almost_empty_if_writing = 1'b0;
    end
    else begin
        assign almost_empty_if_writing = ~occ_low_reset[ALMOST_EMPTY_CUTOFF-1];
    end
    endgenerate
    
    //if writing normally should just be: valid_in & ~stall_out, things get a bit complicated when we have never overflows and write and read during full
    //but these special cases are only relevant when the almost empty cutoff is large enough that valid_in is impacted by fifo full (also note that full asserts during reset)
    generate
    if ((ALMOST_EMPTY_CUTOFF == DEPTH) && NEVER_OVERFLOWS) begin
        assign valid_in_correct_timing_mask = ~occ_in_reset;
    end
    else if ((ALMOST_EMPTY_CUTOFF == DEPTH) && WRITE_AND_READ_DURING_FULL) begin
        assign valid_in_correct_timing_mask = (~occ_in_reset & ~stall_in_correct_timing) | ~occ_high_reset[DEPTH-1];
    end
    else begin
        assign valid_in_correct_timing_mask = ~occ_high_reset[DEPTH-1];
    end
    endgenerate
    
    
    
    
    
    //assign almost_empty_select = valid_in_correct_timing & ~occ_high_reset[DEPTH-1];
    //assign valid_in_correct_timing_mask = ~occ_high_reset[DEPTH-1] | (~occ_in_reset & ~stall_in_correct_timing);
    
    assign almost_empty_pre = (valid_in_correct_timing & valid_in_correct_timing_mask) ? almost_empty_if_writing : almost_empty_if_not_writing;
    
    //same conversion of write_used_words to read_used_words
    //valid_out = (read_used_words != 0), and that can only happen if there is an incoming write or if write_used_words != 0
    assign valid_out_pre = (valid_in_correct_timing & ~occ_high_reset[DEPTH-1]) | occ_low_reset[0];
    
    //retime almost_empty and valid_out
    generate
    if (EARLINESS == 1) begin
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                almost_empty <= 1'b1;
                valid_out <= 1'b0;
                forced_read_out <= 1'b0;
            end
            else begin
                almost_empty <= almost_empty_pre;
                valid_out <= valid_out_pre;
                forced_read_out <= valid_out_pre & ~stall_in_correct_timing;
                if (~sclrn) begin
                    almost_empty <= 1'b1;
                    valid_out <= 1'b0;
                    forced_read_out <= 1'b0;
                end
            end
        end
    end
    else begin  //EARLINESS == 0
        assign almost_empty = almost_empty_pre;
        assign valid_out = valid_out_pre;
        assign forced_read_out = valid_out & ~stall_in_correct_timing;
    end
    endgenerate
    
    
    //output data, take it from the low latency fifo if it has data (this keeps stuff in order), otherwise the fifo is empty so use the data bypass
    //when holding the output data, the HOLD_DATA_OUT_WHEN_EMPTY parameter to the low latency fifo does not work when data is read directly from the bypass
    //because in this case it never got written into the low latency fifo (and writing it causes lots of complications)
    generate
    if (!HOLD_DATA_OUT_WHEN_EMPTY) begin
        assign data_out = (fifo_valid_out) ? fifo_data_out : data_in;
    end
    else begin
        logic data_sel_pri_pre, data_sel_sec_pre, data_sel_pri, data_sel_sec;
        logic captured_data_clock_en_pre, captured_data_clock_en;
        logic [WIDTH-1:0] captured_data_out;
        
        assign data_sel_pri_pre = occ_low_reset[0];
        assign data_sel_sec_pre = valid_in_correct_timing;
        assign captured_data_clock_en_pre = valid_out_pre & ~stall_in_correct_timing;
        
        if (EARLINESS) begin
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_sel_pri <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_sel_sec <= (RESET_EVERYTHING) ? '0 : 'x;
                    captured_data_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    data_sel_pri <= data_sel_pri_pre;
                    data_sel_sec <= data_sel_sec_pre;
                    captured_data_clock_en <= captured_data_clock_en_pre;
                    if (~sclrn && RESET_EVERYTHING) begin
                        data_sel_pri <= 1'b0;
                        data_sel_sec <= 1'b0;
                        captured_data_clock_en <= 1'b0;
                    end
                end
            end
        end
        else begin
            assign data_sel_pri = data_sel_pri_pre;
            assign data_sel_sec = data_sel_sec_pre;
            assign captured_data_clock_en = captured_data_clock_en_pre;
        end
        
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                captured_data_out <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                if (captured_data_clock_en) captured_data_out <= data_out;
                if (~sclrn && RESET_EVERYTHING) captured_data_out <= '0;
            end
        end
        
        assign data_out = (data_sel_pri) ? fifo_data_out : (data_sel_sec) ? data_in : captured_data_out;
    end
    endgenerate
    
    
endmodule

`default_nettype wire
