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
//  ACL LOW LATENCY FIFO                                                                                                                    //
//  Designed and optimized by: Jason Thong                                                                                                  //
//                                                                                                                                          //
//  DESCRIPTION                                                                                                                             //
//  ===========                                                                                                                             //
//  This fifo is fmax optimized for Stratix 10 at the expense of area. This fifo uses registers as storage and therefore it is not          //
//  recommended for large DEPTH. The low latency designation means that every input affects the output within 1 clock cycle, e.g. the       //
//  write-to-read latency is 1 clock. Stall/valid semantics are used, where upstream indicates availability through valid and downstream    //
//  applies backpressure through stall. A write into the fifo occurs when valid_in==1 && stall_out==0. A read from the fifo occurs when     //
//  valid_out==1 && stall_in==0.                                                                                                            //
//                                                                                                                                          //
//  REQUIRED FILES                                                                                                                          //
//  ==============                                                                                                                          //
//  - acl_low_latency_fifo.sv                                                                                                               //
//  - acl_reset_handler.sv                                                                                                                  //
//  - acl_fanout_pipeline.sv                                                                                                                //
//  - acl_std_synchronizer_nocut.sv                                                                                                         //
//                                                                                                                                          //
//  RESET BEHAVIOR                                                                                                                          //
//  ==============                                                                                                                          //
//  During reset the fifo appears as both full and empty. One can interact with the fifo once full deasserts. The almost_full signal also   //
//  follows the same behavior as full for reset. See acl_high_speed_fifo.sv for typical values that the reset parameters should have for    //
//  various platforms.                                                                                                                      //
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
//  =========================================                                                                                               //
//  This is an fmax optimization. If valid_in and stall_in are known one clock cycle ahead of time, we compute occupancy one clock early.   //
//  This gives an extra clock cycle from the control logic (occupancy, clock enables, mux selects) to the data path logic, which helps      //
//  mitigate fmax degradation due to high fan-out (large WIDTH). Only values 0 and 1 are supported since the biggest fmax gain comes from   //
//  registering the control logic before consumption. Any excess earliness is absorbed by hld_fifo, we leave it to Quartus to retime it.    //
//                                                                                                                                          //
//  Note that NEVER_OVERFLOWS and WRITE_AND_READ_DURING_FULL are two separate features and any combination of the two will work properly.   //
//  NEVER_OVERFLOWS means one must externally track occupancy to ensure valid_in is not asserted when the fifo should not be written to.    //
//  WRITE_AND_READ_DURING_FULL means the fifo cannot be written to when the fifo is full and not being read.                                //
//                                                                                                                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TESTBENCH: the testbench for this has now been merged into testbench of hld_fifo


`default_nettype none

module acl_low_latency_fifo #(
    //basic fifo configuration
    parameter int WIDTH,                        // width of the data path through the fifo
    parameter int DEPTH,                        // capacity of the fifo

    //occupancy
    parameter int ALMOST_EMPTY_CUTOFF = 0,      // almost_empty asserts if read_used_words <= ALMOST_EMPTY_CUTOFF, read_used_words increments when writes are visible on the read side, decrements when fifo is read
    parameter int ALMOST_FULL_CUTOFF = 0,       // almost_full asserts if write_used_words >= (DEPTH-ALMOST_FULL_CUTOFF), write_used_words increments when fifo is written to, decrements when fifo is read
    parameter int INITIAL_OCCUPANCY = 0,        // number of words in the fifo when it comes out of reset

    //reset configuration
    parameter bit ASYNC_RESET = 1,              // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 0,        // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    parameter bit RESET_EVERYTHING = 0,         // intended for partial reconfig debug, set to 1 to reset every register (normally async reset excludes data path and sync reset additionally excludes some control signals)
    parameter bit RESET_EXTERNALLY_HELD = 0,    // has no effect since low latency fifo doesn't need reset pulse stretching, put here for parameter compatibility with hld_fifo

    //special configurations for higher fmax / lower area
    parameter int STALL_IN_EARLINESS = 0,       // how many clock cycles early is stall_in provided, fifo can take advantage of 1 clock or stall_in earliness if valid_in is also at least 1 clock early
    parameter int VALID_IN_EARLINESS = 0,       // how many clock cycles early is valid_in provided, fifo can take advantage of 1 clock or valid_in earliness if stall_in is also at least 1 clock early
    parameter int STALL_IN_OUTSIDE_REGS = 0,    // number of registers on the stall-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int VALID_IN_OUTSIDE_REGS = 0,    // number of registers on the valid-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int REGISTERED_DATA_OUT_COUNT = 0,// 0 to WIDTH inclusive, data_out[REGISTERED_DATA_OUT_COUNT-1:0] are registered, the remaining upper bits are unregistered
                                                // generally REGISTERED_DATA_OUT_COUNT should be 0 unless fifo output data drives control logic, in which case just those bits should be registered
                                                // this parameter is ignored if DEPTH == 1 in which case data_out is always registered
    parameter bit NEVER_OVERFLOWS = 0,          // set to 1 to disable fifo's internal overflow protection, stall_out still asserts during reset but won't mask valid_in

    //special features that typically have an fmax penalty
    parameter bit HOLD_DATA_OUT_WHEN_EMPTY = 0, // 0 means data_out can be x when fifo is empty, 1 means data_out will hold last value when fifo is empty (scfifo behavior, has fmax penalty)
    parameter bit WRITE_AND_READ_DURING_FULL = 0,// set to 1 to allow writing and reading while the fifo is full, this may have an fmax penalty, to compensate it is recommended to use this with NEVER_OVERFLOWS = 1

    //hidden parameters
    parameter int RESET_RELEASE_DELAY_OVERRIDE_FROM_ZERO_LATENCY_FIFO = -1  //do not touch, only acl_zero_latency_fifo should set this
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
    output logic                forced_read_out,// indicates fifo is being read on current clock cycle, read data must be consumed or it will be lost, is a registered signal if STALL_IN_EARLINESS >= 1 && VALID_IN_EARLINESS >= 1

    //expose occupancy to outside world, intended for use by acl_zero_latency_fifo, also potentially useful if one wants multiple thresholds for almost full or empty
    output logic    [DEPTH-1:0] occ,            // occupancy, see description below for the encoding
    output logic    [DEPTH-1:0] occ_low_reset,  // occupancy intended for use in backpressuring downstream e.g. almost empty -- during reset the occupancy encoding of 0 means fifo is empty
    output logic    [DEPTH-1:0] occ_high_reset // occupancy intended for use in backpressuring upstream e.g. almost full -- during reset the occupancy encoding of all ones means fifo is full
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
        $fatal(1, "acl_low_latency_fifo: illegal value of DEPTH = %d, minimum allowed is 1\n", DEPTH);
    end
    if ((ALMOST_EMPTY_CUTOFF < 0) || (ALMOST_EMPTY_CUTOFF > DEPTH)) begin
        $fatal(1, "acl_low_latency_fifo: illegal value of ALMOST_EMPTY_CUTOFF = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", ALMOST_EMPTY_CUTOFF, DEPTH);
    end
    if ((ALMOST_FULL_CUTOFF < 0) || (ALMOST_FULL_CUTOFF > DEPTH)) begin
        $fatal(1, "acl_low_latency_fifo: illegal value of ALMOST_FULL_CUTOFF = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", ALMOST_FULL_CUTOFF, DEPTH);
    end
    if ((INITIAL_OCCUPANCY < 0) || (INITIAL_OCCUPANCY > DEPTH)) begin
        $fatal(1, "acl_low_latency_fifo: illegal value of INITIAL_OCCUPANCY = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", INITIAL_OCCUPANCY, DEPTH);
    end
    if ((REGISTERED_DATA_OUT_COUNT < 0) || (REGISTERED_DATA_OUT_COUNT > WIDTH)) begin
        $fatal(1, "acl_low_latency_fifo: illegal value of REGISTERED_DATA_OUT_COUNT = %d, minimum allowed is 0, maximum allowed is WIDTH = %d\n", REGISTERED_DATA_OUT_COUNT, WIDTH);
    end
    if ((STALL_IN_EARLINESS < 0) || (STALL_IN_EARLINESS > 10)) begin
        $fatal(1, "acl_low_latency_fifo: illegal value of STALL_IN_EARLINESS = %d, minimum allowed is 0, maximum allowed is 10\n", STALL_IN_EARLINESS);
    end
    if ((VALID_IN_EARLINESS < 0) || (VALID_IN_EARLINESS > 10)) begin
        $fatal(1, "acl_low_latency_fifo: illegal value of VALID_IN_EARLINESS = %d, minimum allowed is 0, maximum allowed is 10\n", VALID_IN_EARLINESS);
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
    localparam int RESET_RELEASE_DELAY_PRE  = (RAW_RESET_RELEASE_DELAY < MIN_RESET_RELEASE_DELAY) ? MIN_RESET_RELEASE_DELAY : RAW_RESET_RELEASE_DELAY;  //how many clocks late the fifo exits from safe state
    localparam int RESET_RELEASE_DELAY      = (RESET_RELEASE_DELAY_OVERRIDE_FROM_ZERO_LATENCY_FIFO != -1) ? RESET_RELEASE_DELAY_OVERRIDE_FROM_ZERO_LATENCY_FIFO : RESET_RELEASE_DELAY_PRE;

    // properties of the fifo which are consumed by the testbench
    localparam int WRITE_TO_READ_LATENCY            = 1;    //once something is written into the fifo, how many clocks later will it be visible on the read side
    localparam int RESET_EXT_HELD_LENGTH            = 1;    //how many clocks does reset need to be held for, this fifo does not take advantage of RESET_EXTERNALLY_HELD
    localparam int MAX_CLOCKS_TO_ENTER_SAFE_STATE   = 5;    //upon assertion of reset, worse case number of clocks until fifo shows both full and empty
    localparam int MAX_CLOCKS_TO_EXIT_SAFE_STATE    = 15;   //upon release of reset, worse case number of clocks until fifo is ready to transact (not necessarily observable if INITIAL_OCCUPANCY = DEPTH)



    ///////////////////////////
    //                       //
    //  Signal declarations  //
    //                       //
    ///////////////////////////

    genvar            g;
    logic             aclrn, sclrn_pre, sclrn, sclrn_reset_everything; //reset
    logic [RESET_RELEASE_DELAY:0] resetn_delayed;
    logic             fifo_in_reset;

    //retime stall_in and valid_in to the correct timing, absorb excess earliness that the fifo cannot take advantage of
    logic stall_in_correct_timing, valid_in_correct_timing;
    logic [EXCESS_EARLY_STALL:0] stall_in_pipe;
    logic [EXCESS_EARLY_VALID:0] valid_in_pipe;

    //occupancy tracking
    logic [DEPTH-1:0] occ_next;

    //depth 1 (and larger) fifo
    logic             data_out_reg_clock_en, data_out_reg_clock_en_pre;

    //depth 2 (and larger) fifo
    //signals for registered output data
    logic [WIDTH-1:0] data_out_reg, data_one_reg;
    logic             data_out_reg_select, data_out_reg_select_pre;
    logic             data_one_reg_clock_en, data_one_reg_clock_en_pre;
    //signals for unregistered output data
    logic             stall_in_prev, full;
    logic [WIDTH-1:0] data_out_unreg, data_zero_unreg, data_one_unreg;
    logic             data_out_unreg_select, data_out_unreg_select_pre;
    logic             data_zero_unreg_clock_en, data_zero_unreg_clock_en_pre;
    logic             data_zero_unreg_clock_en_pre_default, data_zero_unreg_clock_en_pre_hold, data_zero_unreg_clock_en_pre_nvof;
    logic             data_zero_unreg_select, data_zero_unreg_select_pre;
    logic             data_one_unreg_clock_en, data_one_unreg_clock_en_pre;
    logic             data_one_unreg_select, data_one_unreg_select_pre;

    //depth 3 (and larger) fifo
    //signals for registered output data
    logic             late_stall_in;
    logic [WIDTH-1:0] data_reg [DEPTH-1:0];
    logic [DEPTH-1:0] data_reg_clock_en, data_reg_clock_en_pre;
    logic [DEPTH-1:0] data_reg_select, data_reg_select_pre;
    logic             data_reg_last_select;
    //signals for unregistered output data
    logic [WIDTH-1:0] data_unreg [DEPTH-1:0];
    logic [DEPTH-1:0] data_unreg_clock_en, data_unreg_clock_en_pre;
    logic [DEPTH-1:0] data_unreg_select, data_unreg_select_pre;
    logic             data_unreg_last_select;



    /////////////
    //         //
    //  Reset  //
    //         //
    /////////////

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
        .o_resetn_synchronized  (),
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
        if (~aclrn) fifo_in_reset <= 1'b1;
        else begin
            fifo_in_reset <= 1'b0;
            if (~resetn_delayed[RESET_RELEASE_DELAY]) fifo_in_reset <= 1'b1;
        end
    end

    logic resetn_host_mask;
    logic [DEPTH-1:0] resetn_mask;
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            resetn_host_mask <= 1'b0;
        end
        else begin
            resetn_host_mask <= 1'b1;
            if (~resetn_delayed[RESET_RELEASE_DELAY-EARLINESS]) resetn_host_mask <= 1'b0;
        end
    end
    always_comb begin
        for (int i=0; i<DEPTH; i++) begin : GEN_RANCOM_BLOCK_NAME_R29
            resetn_mask[i] = (i==INITIAL_OCCUPANCY || (i+1)==INITIAL_OCCUPANCY) ? resetn_host_mask : 1'b1;
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
    assign stall_in_correct_timing = stall_in_pipe[EXCESS_EARLY_STALL];
    assign valid_in_correct_timing = valid_in_pipe[EXCESS_EARLY_VALID];



    //////////////////////////
    //                      //
    //  Occupancy tracking  //
    //                      //
    //////////////////////////

    // occ[i] means that the occupancy is greater than i, e.g. occ[0] means there is 1 or more items
    // occ is retimed early like valid_in and stall_in when EARLINESS == 0
    //
    // The following table lists all the values that occ can take when DEPTH = 4
    //
    // value of occ     | actual occupancy
    // -----------------+-----------------
    // 4'b0000          | 0
    // 4'b0001          | 1
    // 4'b0011          | 2
    // 4'b0111          | 3
    // 4'b1111          | 4

    generate
    if (DEPTH == 1) begin : gen_depth_1_occ
        if (WRITE_AND_READ_DURING_FULL) begin
            assign occ_next[0] = (valid_in_correct_timing & resetn_host_mask) ? 1'b1 : (occ[0] & ~stall_in_correct_timing & resetn_host_mask) ? 1'b0 : occ[0];
        end
        else begin
            assign occ_next[0] = (valid_in_correct_timing & ~occ[0] & resetn_host_mask) ? 1'b1 : (occ[0] & ~stall_in_correct_timing & resetn_host_mask) ? 1'b0 : occ[0];
        end
    end
    else begin : gen_depth_n_occ
        always_comb begin
            occ_next[0] = (valid_in_correct_timing & resetn_mask[0]) ? 1'b1 : (~stall_in_correct_timing & resetn_mask[0]) ? occ[1] : occ[0];
            for (int i=1; i<DEPTH-1; i++) begin   //middle registers have no boundary effect, used when DEPTH >= 3
                occ_next[i] = (valid_in_correct_timing & stall_in_correct_timing & resetn_mask[i]) ? occ[i-1] : (~valid_in_correct_timing & ~stall_in_correct_timing & resetn_mask[i]) ? occ[i+1] : occ[i];
            end
            if (WRITE_AND_READ_DURING_FULL) begin
                occ_next[DEPTH-1] = (~stall_in_correct_timing & ~valid_in_correct_timing & resetn_mask[DEPTH-1]) ? 1'b0 : (valid_in_correct_timing & stall_in_correct_timing & resetn_mask[DEPTH-1]) ? occ[DEPTH-2] : occ[DEPTH-1];
            end
            else begin
                occ_next[DEPTH-1] = (~stall_in_correct_timing & resetn_mask[DEPTH-1]) ? 1'b0 : (valid_in_correct_timing & resetn_mask[DEPTH-1]) ? occ[DEPTH-2] : occ[DEPTH-1];
            end
        end
    end
    endgenerate

    //most of occ_low_reset and occ_high_reset will get trimmed away by Quartus
    //occ_high_reset is needed to apply backpressure to upstream during reset (stall_out and almost_full)
    //occ_low_reset is needed to starve downstream during reset, especially when initial occupancy is nonzero (valid_out and almost_empty)
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            for (int i=0; i<DEPTH; i++) occ[i] <= (i >= INITIAL_OCCUPANCY) ? 1'b0 : 1'b1;
            occ_low_reset <= '0;
            occ_high_reset <= '1;
        end
        else begin
            occ <= occ_next;
            occ_low_reset <= occ_next;
            occ_high_reset <= occ_next;
            if (~resetn_delayed[RESET_RELEASE_DELAY-EARLINESS]) begin   //when EARLINESS=1, exit from reset one clock early so that we get correct values on valid_out and stall_out
                for (int i=0; i<DEPTH; i++) occ[i] <= (i >= INITIAL_OCCUPANCY) ? 1'b0 : 1'b1;
                occ_low_reset <= '0;
                occ_high_reset <= '1;
            end
        end
    end



    ///////////////////////////////////////////////////
    //                                               //
    //  Output control signals related to occupancy  //
    //                                               //
    ///////////////////////////////////////////////////

    //delay by 1 clock before exposing to outside world if we have earliness
    generate
    if (EARLINESS == 1) begin
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                valid_out <= 1'b0;
                stall_out <= 1'b1;
                almost_empty <= 1'b1;
                almost_full <= 1'b1;
                forced_read_out <= 1'b0;
            end
            else begin
                valid_out <= occ_low_reset[0];
                stall_out <= (NEVER_OVERFLOWS) ? 1'b0 : (WRITE_AND_READ_DURING_FULL) ? occ_high_reset[DEPTH-1] & stall_in_correct_timing : occ_high_reset[DEPTH-1];
                almost_empty <= (ALMOST_EMPTY_CUTOFF >= DEPTH) ? 1'b1 : (ALMOST_EMPTY_CUTOFF < 0) ? 1'b0 : ~occ_low_reset[ALMOST_EMPTY_CUTOFF];
                almost_full <= (ALMOST_FULL_CUTOFF >= DEPTH) ? 1'b1 : (ALMOST_FULL_CUTOFF < 0) ? 1'b0 : occ_high_reset[DEPTH-ALMOST_FULL_CUTOFF-1];
                forced_read_out <= occ_low_reset[0] & ~stall_in_correct_timing;
                if (~resetn_delayed[RESET_RELEASE_DELAY]) begin
                    valid_out <= 1'b0;
                    stall_out <= 1'b1;
                    almost_empty <= 1'b1;
                    almost_full <= 1'b1;
                    forced_read_out <= 1'b0;
                end
            end
        end
    end
    else begin  //EARLINESS == 0
        always_comb begin
            valid_out = occ_low_reset[0];
            stall_out = (NEVER_OVERFLOWS) ? fifo_in_reset : (WRITE_AND_READ_DURING_FULL) ? fifo_in_reset | (occ_high_reset[DEPTH-1] & stall_in_correct_timing) : occ_high_reset[DEPTH-1];
            almost_empty = (ALMOST_EMPTY_CUTOFF >= DEPTH) ? 1'b1 : (ALMOST_EMPTY_CUTOFF < 0) ? 1'b0 : ~occ_low_reset[ALMOST_EMPTY_CUTOFF];
            almost_full = (ALMOST_FULL_CUTOFF >= DEPTH) ? 1'b1 : (ALMOST_FULL_CUTOFF < 0) ? 1'b0 : occ_high_reset[DEPTH-ALMOST_FULL_CUTOFF-1];
            forced_read_out = occ_low_reset[0] & ~stall_in_correct_timing;
        end
    end
    endgenerate



    //helper signals
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            stall_in_prev <= (RESET_EVERYTHING) ? '0 : 'x;
            full <= (RESET_EVERYTHING) ? '0 : 'x;
            late_stall_in <= (RESET_EVERYTHING) ? '0 : 'x;
        end
        else begin
            stall_in_prev <= stall_in_correct_timing;
            full <= (EARLINESS == 1) ? occ[DEPTH-1] : occ_next[DEPTH-1];
            late_stall_in <= (EARLINESS == 1) ? stall_in_prev : stall_in_correct_timing;
            if (~sclrn_reset_everything) begin
                stall_in_prev <= '0;
                full <= '0;
                late_stall_in <= '0;
            end
        end
    end



    ////////////////////
    //                //
    //  Depth 1 fifo  //
    //                //
    ////////////////////

    //naming convention: the ****_pre signals are retimed earlier when EARLINESS == 1, this applies to depth 1, depth 2, and depth 3+
    generate
    if (DEPTH == 1) begin : gen_depth_1
        //clock enable
        assign data_out_reg_clock_en_pre = (WRITE_AND_READ_DURING_FULL) ? ((NEVER_OVERFLOWS) ? valid_in_correct_timing : valid_in_correct_timing & (~stall_in_correct_timing | ~occ[0]))  :
            (HOLD_DATA_OUT_WHEN_EMPTY) ? ~occ[0] & valid_in_correct_timing : ~occ[0];

        //retime if we have earliness
        if (EARLINESS == 1) begin
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_out_reg_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    data_out_reg_clock_en <= data_out_reg_clock_en_pre;
                    if (~sclrn_reset_everything) data_out_reg_clock_en <= '0;
                end
            end
        end
        else begin
            always_comb begin
                data_out_reg_clock_en = data_out_reg_clock_en_pre;
            end
        end

        //output data is always registered
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_out_reg <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                if (data_out_reg_clock_en) data_out_reg <= data_in;
                if (~sclrn_reset_everything) data_out_reg <= '0;
            end
        end
        assign data_out_unreg = data_out_reg;
    end
    endgenerate



    ////////////////////
    //                //
    //  Depth 2 fifo  //
    //                //
    ////////////////////

    //generate both registered and unregistered output data versions
    //we can select which version to consume per bit of data path, the unused parts will be synthesized away
    generate
    if (DEPTH == 2) begin : gen_depth_2

        ////////////////////////////////////
        // Registered output data version //
        ////////////////////////////////////

        //clock enable
        assign data_out_reg_clock_en_pre = (HOLD_DATA_OUT_WHEN_EMPTY) ? (~stall_in_correct_timing & (occ[1] | valid_in_correct_timing)) | (~occ[0] & valid_in_correct_timing) : ~stall_in_correct_timing | ~occ[0];
        assign data_one_reg_clock_en_pre = (WRITE_AND_READ_DURING_FULL) ? ~occ[1] | ~stall_in_correct_timing : ~occ[1];

        //mux select
        assign data_out_reg_select_pre = ~occ[1];

        //retime if we have earliness
        if (EARLINESS == 1) begin
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_out_reg_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_out_reg_select <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_one_reg_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    data_out_reg_clock_en <= data_out_reg_clock_en_pre;
                    data_out_reg_select <= data_out_reg_select_pre;
                    data_one_reg_clock_en <= data_one_reg_clock_en_pre;
                    if (~sclrn_reset_everything) begin
                        data_out_reg_clock_en <= '0;
                        data_out_reg_select <= '0;
                        data_one_reg_clock_en <= '0;
                    end
                end
            end
        end
        else begin
            always_comb begin
                data_out_reg_clock_en = data_out_reg_clock_en_pre;
                data_out_reg_select = data_out_reg_select_pre;
                data_one_reg_clock_en = data_one_reg_clock_en_pre;
            end
        end

        //data path
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_out_reg <= (RESET_EVERYTHING) ? '0 : 'x;
                data_one_reg <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                if (data_out_reg_clock_en) data_out_reg <= (data_out_reg_select) ? data_in : data_one_reg;
                if (data_one_reg_clock_en) data_one_reg <= data_in;
                if (~sclrn_reset_everything) begin
                    data_out_reg <= '0;
                    data_one_reg <= '0;
                end
            end
        end


        //////////////////////////////////////
        // Unregistered output data version //
        //////////////////////////////////////

        //clock enable
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_zero_unreg_clock_en_pre_default <= (RESET_EVERYTHING) ? '0 : 'x;
                data_one_unreg_clock_en_pre <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                data_zero_unreg_clock_en_pre_default <= ~stall_in_correct_timing | ~occ[0];
                data_one_unreg_clock_en_pre <= (WRITE_AND_READ_DURING_FULL) ? ~occ_next[1] | ~stall_in_correct_timing : ~occ_next[1];
                if (~sclrn_reset_everything) begin
                    data_zero_unreg_clock_en_pre_default <= '0;
                    data_one_unreg_clock_en_pre <= '0;
                end
            end
        end
        assign data_zero_unreg_clock_en_pre_hold = (HOLD_DATA_OUT_WHEN_EMPTY) ? (valid_in_correct_timing | occ[0]) : 1'b1;
        assign data_zero_unreg_clock_en_pre_nvof = (WRITE_AND_READ_DURING_FULL) ? ((HOLD_DATA_OUT_WHEN_EMPTY) ? ~stall_in_correct_timing & occ[1] : ~stall_in_correct_timing) : 1'b0;
        assign data_zero_unreg_clock_en_pre = (data_zero_unreg_clock_en_pre_default & data_zero_unreg_clock_en_pre_hold) | data_zero_unreg_clock_en_pre_nvof;

        //mux select
        assign data_out_unreg_select_pre = (HOLD_DATA_OUT_WHEN_EMPTY) ? ~stall_in_prev & occ[0] : ~stall_in_prev;
        assign data_zero_unreg_select_pre = ~occ[0] | data_zero_unreg_clock_en_pre_nvof;

        //retime if we have earliness
        if (EARLINESS == 1) begin
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_out_unreg_select <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_zero_unreg_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_zero_unreg_select <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_one_unreg_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    data_out_unreg_select <= data_out_unreg_select_pre;
                    data_zero_unreg_clock_en <= data_zero_unreg_clock_en_pre;
                    data_zero_unreg_select <= data_zero_unreg_select_pre;
                    data_one_unreg_clock_en <= data_one_unreg_clock_en_pre;
                    if (~sclrn_reset_everything) begin
                        data_out_unreg_select <= '0;
                        data_zero_unreg_clock_en <= '0;
                        data_zero_unreg_select <= '0;
                        data_one_unreg_clock_en <= '0;
                    end
                end
            end
        end
        else begin
            always_comb begin
                data_out_unreg_select = data_out_unreg_select_pre;
                data_zero_unreg_clock_en = data_zero_unreg_clock_en_pre;
                data_zero_unreg_select = data_zero_unreg_select_pre;
                data_one_unreg_clock_en = data_one_unreg_clock_en_pre;
            end
        end

        //late mux select, do not register so that constant is easily seen when WRITE_AND_READ_DURING_FULL = 0
        assign data_one_unreg_select = (WRITE_AND_READ_DURING_FULL) ? ~full : 1'b1;

        //data path
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_zero_unreg <= (RESET_EVERYTHING) ? '0 : 'x;
                data_one_unreg <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                if (data_zero_unreg_clock_en) data_zero_unreg <= (data_zero_unreg_select) ? data_in : data_one_unreg;
                if (data_one_unreg_clock_en) data_one_unreg <= (data_one_unreg_select) ? data_in : data_zero_unreg;
                if (~sclrn_reset_everything) begin
                    data_zero_unreg <= '0;
                    data_one_unreg <= '0;
                end
            end
        end
        assign data_out_unreg = (data_out_unreg_select) ? data_one_unreg : data_zero_unreg;
    end
    endgenerate



    /////////////////////
    //                 //
    //  Depth 3+ fifo  //
    //                 //
    /////////////////////

    //generate both registered and unregistered output data versions
    //we can select which version to consume per bit of data path, the unused parts will be synthesized away
    generate
    if (DEPTH >= 3) begin : gen_depth_3_plus

        ////////////////////////////////////
        // Registered output data version //
        ////////////////////////////////////

        //clock enable
        always_comb begin
            data_reg_clock_en_pre[0] = (HOLD_DATA_OUT_WHEN_EMPTY) ? (~stall_in_correct_timing & (occ[1] | valid_in_correct_timing)) | (~occ[0] & valid_in_correct_timing) : ~stall_in_correct_timing | ~occ[0];
        end
        for (g=1; g<DEPTH-1; g++) begin : gen_data_reg_clock_en_pre
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_reg_clock_en_pre[g] <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    data_reg_clock_en_pre[g] <= ~stall_in_correct_timing | ~occ[g];
                    if (~sclrn_reset_everything) data_reg_clock_en_pre[g] <= '0;
                end
            end
        end
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_reg_clock_en_pre[DEPTH-1] <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                data_reg_clock_en_pre[DEPTH-1] <= (WRITE_AND_READ_DURING_FULL) ? ~occ_next[DEPTH-1] | ~stall_in_correct_timing : ~occ_next[DEPTH-1];
                if (~sclrn_reset_everything) data_reg_clock_en_pre[DEPTH-1] <= '0;
            end
        end

        //mux select
        always_comb begin
            data_reg_select_pre = ~occ;
            data_reg_select_pre[0] = ~occ[1];
            if (WRITE_AND_READ_DURING_FULL) data_reg_select_pre[1] = ~occ[1] | ~stall_in_correct_timing;
        end

        //retime if we have earliness
        if (EARLINESS == 1) begin
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_reg_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_reg_select <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    data_reg_clock_en <= data_reg_clock_en_pre;
                    data_reg_select <= data_reg_select_pre;
                    if (WRITE_AND_READ_DURING_FULL) data_reg_clock_en[1] <= data_reg_clock_en_pre[1] | ~stall_in_correct_timing;
                    if (~sclrn_reset_everything) begin
                        data_reg_clock_en <= '0;
                        data_reg_select <= '0;
                    end
                end
            end
        end
        else begin
            always_comb begin
                data_reg_clock_en = data_reg_clock_en_pre;
                data_reg_select = data_reg_select_pre;
                if (WRITE_AND_READ_DURING_FULL) data_reg_clock_en[1] = data_reg_clock_en_pre[1] | ~stall_in_correct_timing;
            end
        end

        //late mux select, do not register so that constant is easily seen when WRITE_AND_READ_DURING_FULL = 0
        assign data_reg_last_select = (WRITE_AND_READ_DURING_FULL) ? ~full : 1'b1;

        //data path
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_reg[0] <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                if (data_reg_clock_en[0]) data_reg[0] <= (data_reg_select[0]) ? data_in : (late_stall_in) ? data_reg[1] : data_reg[2];
                if (~sclrn_reset_everything) data_reg[0] <= '0;
            end
        end
        for (g=1; g<DEPTH-1; g++) begin : gen_data_reg
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_reg[g] <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    if (data_reg_clock_en[g]) data_reg[g] <= (data_reg_select[g]) ? data_in : data_reg[g+1];
                    if (~sclrn_reset_everything) data_reg[g] <= '0;
                end
            end
        end
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_reg[DEPTH-1] <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                if (data_reg_clock_en[DEPTH-1]) data_reg[DEPTH-1] <= (data_reg_last_select) ? data_in : data_reg[1];
                if (~sclrn_reset_everything) data_reg[DEPTH-1] <= '0;
            end
        end
        assign data_out_reg = data_reg[0];


        //////////////////////////////////////
        // Unregistered output data version //
        //////////////////////////////////////

        //clock enable
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_unreg_clock_en_pre <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                for (int i=0; i<DEPTH-1; i++) begin : GEN_RANDOM_BLOCK_NAME_R30
                    data_unreg_clock_en_pre[i] <= ~stall_in_correct_timing | ~occ[i];
                end
                data_unreg_clock_en_pre[DEPTH-1] <= ~occ_next[DEPTH-1] | ((WRITE_AND_READ_DURING_FULL) ? ~stall_in_correct_timing : 1'b0);
                if (~sclrn_reset_everything) data_unreg_clock_en_pre <= '0;
            end
        end
        assign data_zero_unreg_clock_en_pre_hold = (HOLD_DATA_OUT_WHEN_EMPTY) ? (valid_in_correct_timing | occ[0]) : 1'b1;
        assign data_zero_unreg_clock_en_pre_nvof = (WRITE_AND_READ_DURING_FULL) ? ((HOLD_DATA_OUT_WHEN_EMPTY) ? ~stall_in_correct_timing & occ[1] : ~stall_in_correct_timing) : 1'b0;
        assign data_zero_unreg_clock_en_pre = (data_unreg_clock_en_pre[0] & data_zero_unreg_clock_en_pre_hold) | data_zero_unreg_clock_en_pre_nvof;

        //mux select
        always_comb begin
            data_unreg_select_pre = ~occ;
            data_unreg_select_pre[0] = ~occ[0] | data_zero_unreg_clock_en_pre_nvof;
            data_out_unreg_select_pre = (!HOLD_DATA_OUT_WHEN_EMPTY) ? (~stall_in_prev) : (~stall_in_prev & occ[0]);
        end

        //retime if we have earliness
        if (EARLINESS == 1) begin
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_unreg_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_unreg_select <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_out_unreg_select <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    data_unreg_clock_en <= data_unreg_clock_en_pre;
                    data_unreg_clock_en[0] <= data_zero_unreg_clock_en_pre;
                    data_unreg_select <= data_unreg_select_pre;
                    data_out_unreg_select <= data_out_unreg_select_pre;
                    if (~sclrn_reset_everything) begin
                        data_unreg_clock_en <= '0;
                        data_unreg_select <= '0;
                        data_out_unreg_select <= '0;
                    end
                end
            end
        end
        else begin
            always_comb begin
                data_unreg_clock_en = data_unreg_clock_en_pre;
                data_unreg_clock_en[0] = data_zero_unreg_clock_en_pre;
                data_unreg_select = data_unreg_select_pre;
                data_out_unreg_select = data_out_unreg_select_pre;
            end
        end

        //late mux select, do not register so that constant is easily seen when WRITE_AND_READ_DURING_FULL = 0
        assign data_unreg_last_select = (WRITE_AND_READ_DURING_FULL) ? ~full : 1'b1;

        //data path
        for (g=0; g<DEPTH-1; g++) begin : gen_data_unreg
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_unreg[g] <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    if (data_unreg_clock_en[g]) data_unreg[g] <= (data_unreg_select[g]) ? data_in : data_unreg[g+1];
                    if (~sclrn_reset_everything) data_unreg[g] <= '0;
                end
            end
        end
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                data_unreg[DEPTH-1] <= (RESET_EVERYTHING) ? '0 : 'x;
            end
            else begin
                if (data_unreg_clock_en[DEPTH-1]) data_unreg[DEPTH-1] <= (data_unreg_last_select) ? data_in : data_unreg[0];
                if (~sclrn_reset_everything) data_unreg[DEPTH-1] <= '0;
            end
        end
        assign data_out_unreg = (data_out_unreg_select) ? data_unreg[1] : data_unreg[0];
    end
    endgenerate



    ///////////////////////////////////////////////////////////////////
    //                                                               //
    //  Select whether to use registered or unregistered output data //
    //                                                               //
    ///////////////////////////////////////////////////////////////////

    generate
    if (REGISTERED_DATA_OUT_COUNT == 0) begin : gen_unregistered_data_out
        assign data_out = data_out_unreg;
    end
    else if (REGISTERED_DATA_OUT_COUNT == WIDTH) begin : gen_registered_data_out
        assign data_out = data_out_reg;
    end
    else begin : gen_mixed_reg_data_out
        assign data_out = {data_out_unreg[WIDTH-1:REGISTERED_DATA_OUT_COUNT], data_out_reg[REGISTERED_DATA_OUT_COUNT-1:0]};
    end
    endgenerate

endmodule

`default_nettype wire
