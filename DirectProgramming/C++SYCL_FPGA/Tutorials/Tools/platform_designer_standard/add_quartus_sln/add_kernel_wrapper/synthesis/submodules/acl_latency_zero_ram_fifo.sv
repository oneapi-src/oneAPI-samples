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


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                               //
//  ACL LATENCY ZERO RAM FIFO                                                                                                                                                                                    //
//  Designed and optimized by: Jason Thong                                                                                                                                                                       //
//                                                                                                                                                                                                               //
//  DESCRIPTION                                                                                                                                                                                                  //
//  ===========                                                                                                                                                                                                  //
//  This fifo has a write to read latency of zero clock cycles, e.g. on the same clock cycle that data is written it is also readable. Zero latency is a bit of a tricky concept in that read_used_words         //
//  changes BEFORE write_used_words. The interpretation is that data can bypass the storage associated with a fifo e.g. data does not need to be written to a register or a memory in order to be readable.      //
//                                                                                                                                                                                                               //
//  REQUIRED FILES                                                                                                                                                                                               //
//  ==============                                                                                                                                                                                               //
//  - acl_latency_zero_ram_fifo.sv                                                                                                                                                                               //
//  - acl_latency_one_ram_fifo.sv                                                                                                                                                                                //
//  - acl_lfsr.sv                                                                                                                                                                                                //
//  - acl_tessellated_incr_decr_threshold.sv                                                                                                                                                                     //
//  - acl_reset_handler.sv                                                                                                                                                                                       //
//                                                                                                                                                                                                               //
//  RELATIONSHIP TO ACL_LATENCY_ONE_RAM_FIFO                                                                                                                                                                     //
//  ========================================                                                                                                                                                                     //
//  This fifo is built on top of acl_latency_one_ram_fifo with a combinational logic bypass so that incoming data can be forwarded to the output on the same clock cycle. Therefore it shares the same limits    //
//  for maximum earliness, and it shares the caveats inherited all the way from acl_mid_speed_fifo like different implementations for MLAB and M20K.                                                             //
//                                                                                                                                                                                                               //
//  RELATIONSHIP TO ACL_ZERO_LATENCY_FIFO                                                                                                                                                                        //
//  =====================================                                                                                                                                                                        //
//  Functionally, this fifo can be used interchangeably with acl_zero_latency_fifo. The main difference is the logic utilization. This fifo stores data in memory whereas acl_zero_latency_fifo uses registers.  // 
//  Just like acl_latency_one_ram_fifo vs acl_low_latency_fifo, at shallow depths it is better to use registers, and as the capacity increases eventually memory results in lower logic utilization.             //
//                                                                                                                                                                                                               //
//  HLD_FIFO FEATURES                                                                                                                                                                                            //
//  =================                                                                                                                                                                                            //
//  This fifo is fully feature complete against all of the parameters exposed by hld_fifo. It follows the same reset scheme of asserting full and empty during reset. Note that REGISTERED_DATA_OUT_COUNT is     //
//  applied to the underlying acl_latency_one_ram_fifo. Due to the same clock cycle data bypass, it is impossible for a zero latency fifo to ever have registered output data.                                   //
//                                                                                                                                                                                                               //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

`default_nettype none

module acl_latency_zero_ram_fifo #(
    //basic fifo configuration
    parameter int WIDTH,                        // width of the data path through the fifo
    parameter int DEPTH,                        // capacity of the fifo, at least 1
    
    //occupancy
    parameter int ALMOST_EMPTY_CUTOFF = 0,      // almost_empty asserts if read_used_words <= ALMOST_EMPTY_CUTOFF, read_used_words increments when writes are visible on the read side, decrements when fifo is read
    parameter int ALMOST_FULL_CUTOFF = 0,       // almost_full asserts if write_used_words >= (DEPTH-ALMOST_FULL_CUTOFF), write_used_words increments when fifo is written to, decrements when fifo is read
    parameter int INITIAL_OCCUPANCY = 0,        // number of words in the fifo (write side occupancy) when it comes out of reset, note it still takes 5 clocks for this to become visible on the read side
    
    //reset configuration
    parameter bit ASYNC_RESET = 0,              // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 1,        // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    parameter bit RESET_EVERYTHING = 0,         // intended for partial reconfig debug, set to 1 to reset every register (normally async reset excludes data path and sync reset additionally excludes some control signals)
    parameter bit RESET_EXTERNALLY_HELD = 1,    // set to 1 if resetn will be held for at least FOUR clock cycles, otherwise we will internally pulse stretch reset before consumption
    
    //ram implementation
    parameter string RAM_BLOCK_TYPE = "FIFO_TO_CHOOSE", // "MLAB" | "M20K" | "FIFO_TO_CHOOSE" -> different implementations for MLAB vs M20K, so cannot let quartus decide in case MLAB or M20K is not explicitly specified
    
    //special configurations for higher fmax / low area
    parameter int STALL_IN_EARLINESS = 0,       // how many clock cycles early is stall_in provided, fifo supports up to 2, setting this any higher results in registers to absorb the excess earliness
    parameter int VALID_IN_EARLINESS = 0,       // how many clock cycles early is valid_in provided, fifo can take advantage of 1 clock or valid_in earliness if stall_in is also at least 2 clocks early
    parameter int STALL_IN_OUTSIDE_REGS = 0,    // number of registers on the stall-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int VALID_IN_OUTSIDE_REGS = 0,    // number of registers on the valid-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int REGISTERED_DATA_OUT_COUNT = 0,// this is passed to acl_latency_one_ram_fifo, the output of a zero latency fifo can never be registered
    parameter bit NEVER_OVERFLOWS = 0,          // set to 1 to disable fifo's internal overflow protection, area savings by removing one incr/decr/thresh, stall_out still asserts during reset but won't mask valid_in
    
    //special features that typically have an fmax penalty
    parameter bit HOLD_DATA_OUT_WHEN_EMPTY = 0, // 0 means data_out can be x when fifo is empty, 1 means data_out will hold last value when fifo is empty (scfifo behavior, has fmax penalty)
    parameter bit WRITE_AND_READ_DURING_FULL = 0,//set to 1 to allow writing and reading while the fifo is full, this may have an fmax penalty, to compensate it is recommended to use this with NEVER_OVERFLOWS = 1
    
    //error correction code
    parameter enable_ecc = "FALSE"              // NOT IMPLEMENTED YET, see case:555783
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
    output logic                forced_read_out,// indicates fifo is being read on current clock cycle, read data must be consumed or it will be lost, is a registered signal if STALL_IN_EARLINESS >= 1
    
    //other
    output logic          [1:0] ecc_err_status  // NOT IMPLEMENTED YET, see case:555783
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
        $fatal(1, "acl_latency_zero_ram_fifo: illegal value of DEPTH = %d, minimum allowed is 1\n", DEPTH);
    end
    if ((ALMOST_EMPTY_CUTOFF < 0) || (ALMOST_EMPTY_CUTOFF > DEPTH)) begin
        $fatal(1, "acl_latency_zero_ram_fifo: illegal value of ALMOST_EMPTY_CUTOFF = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", ALMOST_EMPTY_CUTOFF, DEPTH);
    end
    if ((ALMOST_FULL_CUTOFF < 0) || (ALMOST_FULL_CUTOFF > DEPTH)) begin
        $fatal(1, "acl_latency_zero_ram_fifo: illegal value of ALMOST_FULL_CUTOFF = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", ALMOST_FULL_CUTOFF, DEPTH);
    end
    if ((INITIAL_OCCUPANCY < 0) || (INITIAL_OCCUPANCY > DEPTH)) begin
        $fatal(1, "acl_latency_zero_ram_fifo: illegal value of INITIAL_OCCUPANCY = %d, minimum allowed is 0, maximum allowed is DEPTH = %d\n", INITIAL_OCCUPANCY, DEPTH);
    end
    if ((REGISTERED_DATA_OUT_COUNT < 0) || (REGISTERED_DATA_OUT_COUNT > WIDTH)) begin
        $fatal(1, "acl_latency_zero_ram_fifo: illegal value of REGISTERED_DATA_OUT_COUNT = %d, minimum allowed is 0, maximum allowed is WIDTH = %d\n", REGISTERED_DATA_OUT_COUNT, WIDTH);
    end
    if ((STALL_IN_EARLINESS < 0) || (STALL_IN_EARLINESS > 10)) begin
        $fatal(1, "acl_latency_zero_ram_fifo: illegal value of STALL_IN_EARLINESS = %d, minimum allowed is 0, maximum allowed is 10\n", STALL_IN_EARLINESS);
    end
    if ((VALID_IN_EARLINESS < 0) || (VALID_IN_EARLINESS > 10)) begin
        $fatal(1, "acl_latency_zero_ram_fifo: illegal value of VALID_IN_EARLINESS = %d, minimum allowed is 0, maximum allowed is 10\n", VALID_IN_EARLINESS);
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
    
    // earliness configuration
    localparam int EARLY_MODE           = (STALL_IN_EARLINESS >= 1 && VALID_IN_EARLINESS >= 1) ? 1 : 0;
    localparam int EARLY_VALID          = EARLY_MODE;
    localparam int EXCESS_EARLY_VALID   = VALID_IN_EARLINESS - EARLY_VALID;
    localparam int EARLY_STALL          = EARLY_MODE;
    localparam int EXCESS_EARLY_STALL   = STALL_IN_EARLINESS - EARLY_STALL;
    
    // reset timing
    localparam int EXCESS_EARLY_STALL_WITH_EXT = EXCESS_EARLY_STALL + STALL_IN_OUTSIDE_REGS;    //early stall is affected by regs outside this module; account for effect on reset timing
    localparam int EXCESS_EARLY_VALID_WITH_EXT = EXCESS_EARLY_VALID + VALID_IN_OUTSIDE_REGS;    //early valid is affected by regs outisde this module; account for effect on reset timing
    localparam int FLUSH_EARLY_PIPES    = (EXCESS_EARLY_STALL_WITH_EXT > EXCESS_EARLY_VALID_WITH_EXT) ? EXCESS_EARLY_STALL_WITH_EXT : EXCESS_EARLY_VALID_WITH_EXT;  // clocks needs to flush excess earliness pipelines
    localparam int RESET_SYNC_DEPTH     = (SYNCHRONIZE_RESET) ? 3 : 0;                                                          // how many registers are added inside acl_reset_handler for synchronizing the reset
    localparam int RESET_PIPE_DEPTH     = 2;                                                                                    // how many pipeline stages we add to sclrn
    localparam int RESET_LATENCY        = (ASYNC_RESET || RESET_EVERYTHING) ? 0 : (RESET_SYNC_DEPTH + RESET_PIPE_DEPTH);        // how many clocks from the resetn input signal until the reset is consumed
    localparam int MIN_RESET_DELAY      = EARLY_MODE;                                                                           // internal occ tracking is 1 clock early when EARLY_MODE=1, 1 clock to propagate to stall_out
    localparam int RAW_RESET_DELAY      = FLUSH_EARLY_PIPES - RESET_LATENCY;                                                    // delay fifo exit from safe state if need more clocks to flush earliness than reset latency
    localparam int RESET_RELEASE_DELAY  = (RAW_RESET_DELAY < MIN_RESET_DELAY) ? MIN_RESET_DELAY : RAW_RESET_DELAY;              // how many clocks late the fifo exits from safe state
    
    // reset release delay for the various occupancy trackers
    localparam int RESET_DELAY_ALMOST_EMPTY = RESET_RELEASE_DELAY - EARLY_MODE;
    localparam int RESET_DELAY_MAX          = RESET_RELEASE_DELAY;
    
    // properties of the fifo which are consumed by the testbench
    localparam int WRITE_TO_READ_LATENCY            = 0;    //once something is written into the fifo, how many clocks later will it be visible on the read side
    localparam int RESET_EXT_HELD_LENGTH            = 4;    //if RESET_EXTERNALLY_HELD = 1, how many clocks does reset need to be held for
    localparam int MAX_CLOCKS_TO_ENTER_SAFE_STATE   = 2;    //upon assertion of reset, worse case number of clocks until fifo shows both full and empty
    localparam int MAX_CLOCKS_TO_EXIT_SAFE_STATE    = 18;   //upon release of reset, worse case number of clocks until fifo is ready to transact (not necessarily observable if INITIAL_OCCUPANCY = DEPTH)
    
    
    
    ///////////////////////////
    //                       //
    //  Signal declarations  //
    //                       //
    ///////////////////////////
    
    // Naming convention: some signals are retimed early, any signal ending with _E has the same earliness as the EARLY_MODE parameter.
    
    //reset
    genvar g;
    logic resetn_synchronized;
    logic aclrn, sclrn;                             //these are the typical active low reset signals that are consumed
    logic sclrn_early_two, sclrn_early;             //helpers for sclrn
    logic [RESET_DELAY_MAX:0] resetn_delayed;       //delayed versions of aclrn or sclrn, consumed by the occupancy trackers
    logic fifo_in_reset;                            //intended primarily for consumption by testbench to know when fifo is in reset
    
    //retime stall_in and valid_in to the correct timing, absorb excess earliness that the fifo cannot take advantage of
    logic stall_in_E, valid_in_E;
    logic [EXCESS_EARLY_STALL:0] stall_in_pipe;
    logic [EXCESS_EARLY_VALID:0] valid_in_pipe;
    
    //signals extracted from acl_latency_one_ram_fifo
    logic [WIDTH-1:0] fifo_data_out;                //data_out from latency one fifo
    logic occ_gte_one_E, stall_out_E;               //early valid_out, early stall_out from latency one fifo
    
    //write bypass logic
    logic fifo_valid_in_E;                          //write enable to latency one fifo
    logic valid_out_E;                              //early valid_out for this fifo
    logic forced_read_out_E;                        //did we read from this fifo
    logic occ_gte_one;                              //if latency one fifo has valid data, then our data_out should use the data_out from latency one fifo
    
    //almost empty -- most of the logic for this is declared inside the ALMOST_EMPTY_CUTOFF != 0 block
    logic almost_empty_E;
    
    
    
    /////////////
    //         //
    //  Reset  //
    //         //
    /////////////
    
    // the reset structure is identical to acl_mid_speed_fifo
    
    // S10 reset specification:
    // S (clocks to enter reset safe state) : 2 for sclrn_early_two to actual register (beware synchronizer takes no time for reset assertion, but it does take time for reset release)
    // P (minimum duration of reset pulse)  : 4 if RESET_EXTERNALLY_HELD = 1, otherwise 1 (we will internally pulse stretch the reset to 4 clocks)
    // D (clocks to exit reset safe state)  : 18 (3 for synchronizer) + (5 for sclrn_early_two to actual register) + (10 for reset release delay for registers that absorb excess earliness)
    
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
        .o_resetn_synchronized  (resetn_synchronized),
        .o_sclrn                (sclrn_early_two)
    );
    
    generate
    if (ASYNC_RESET) begin : async_reset
        assign sclrn = 1'b1;
        assign sclrn_early = 1'b1;
    end
    else begin : sync_reset
        logic [2:0] sclrn_chain;
        always_ff @(posedge clock) begin
            sclrn_chain <= (sclrn_chain << 1) | sclrn_early_two;
            sclrn_early <= (RESET_EXTERNALLY_HELD) ? sclrn_early_two : ((&sclrn_chain) & sclrn_early_two);
            sclrn <= sclrn_early;
        end
    end
    endgenerate
    
    generate
    always_comb begin
        resetn_delayed[0] = (ASYNC_RESET) ? aclrn : sclrn;      //delay 0 = original reset timing
    end
    for (g=1; g<=RESET_DELAY_MAX; g++) begin : gen_resetn_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) resetn_delayed[g] <= '0;
            else begin
                resetn_delayed[g] <= (ASYNC_RESET && g==1) ? 1'b1 : resetn_delayed[g-1];    //partial reconfig requires no d-input consumption of aclr, so resetn_delayed[1] loads 1'b1 if ASYNC_RESET
                if (~sclrn_early) resetn_delayed[g] <= '0;                                  //resetn_delayed goes into reset as the same time as sclrn, since this is registered need to peek one clock ahead of sclrn
            end
        end
    end
    endgenerate
    
    //fifo_in_reset is consumed by the testbench to know whether the fifo is still in reset, can't use stall_out when fifo starts as full
    //unlike other fifos, fifo_in_reset is never exposed on the output, as stall_out comes from acl_latency_one_ram_fifo
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) fifo_in_reset <= 1'b1;
        else begin
            fifo_in_reset <= 1'b0;
            if (~resetn_delayed[RESET_RELEASE_DELAY]) fifo_in_reset <= 1'b1;
        end
    end
    
    
    
    ////////////////////////////////////////////////
    //                                            //
    //  Absorb excess earliness on input signals  //
    //                                            //
    ////////////////////////////////////////////////
    
    generate
    always_comb begin
        stall_in_pipe[0] = stall_in;
        valid_in_pipe[0] = valid_in;
    end
    for (g=1; g<=EXCESS_EARLY_STALL; g++) begin : gen_stall_in_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) stall_in_pipe[g] <= 1'b1;
            else begin
                stall_in_pipe[g] <= stall_in_pipe[g-1];
                if (~sclrn && RESET_EVERYTHING) stall_in_pipe[g] <= 1'b1;
            end
        end
    end
    for (g=1; g<=EXCESS_EARLY_VALID; g++) begin : gen_valid_in_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) valid_in_pipe[g] <= 1'b0;
            else begin
                valid_in_pipe[g] <= valid_in_pipe[g-1];
                if (~sclrn && RESET_EVERYTHING) valid_in_pipe[g] <= 1'b0;
            end
        end
    end
    endgenerate
    assign stall_in_E = stall_in_pipe[EXCESS_EARLY_STALL];
    assign valid_in_E = valid_in_pipe[EXCESS_EARLY_VALID];
    
    
    
    ////////////////////////////////////////
    //                                    //
    //  Instantiate the latency one fifo  //
    //                                    //
    ////////////////////////////////////////
    
    acl_latency_one_ram_fifo
    #(
        .WIDTH                          (WIDTH),
        .DEPTH                          (DEPTH),
        .ALMOST_EMPTY_CUTOFF            (0),    //almost_empty is managed externally because read_used_words changes before write_used_words
        .ALMOST_FULL_CUTOFF             (ALMOST_FULL_CUTOFF),
        .INITIAL_OCCUPANCY              (INITIAL_OCCUPANCY),
        .ASYNC_RESET                    (ASYNC_RESET),
        .SYNCHRONIZE_RESET              (0),    //do not synchronize the reset again, this ensures that e.g. stall_out deasserting upon reset exit happens on the same clock cycle here and well as inside latency one fifo
        .RESET_EVERYTHING               (RESET_EVERYTHING),
        .RESET_EXTERNALLY_HELD          (RESET_EXTERNALLY_HELD),
        .STALL_IN_EARLINESS             (EARLY_MODE),   //we have to absorb excess earliness so that valid_in and stall_in have the same earliness (needed for bypass logic), and that earliness can only be 0 or 1...
        .VALID_IN_EARLINESS             (EARLY_MODE),
        .REGISTERED_DATA_OUT_COUNT      (REGISTERED_DATA_OUT_COUNT),
        .NEVER_OVERFLOWS                (NEVER_OVERFLOWS),
        .HOLD_DATA_OUT_WHEN_EMPTY       (0),
        .WRITE_AND_READ_DURING_FULL     (WRITE_AND_READ_DURING_FULL),
        .ZLRAM_RESET_RELEASE_DELAY_OVERRIDE(RESET_RELEASE_DELAY),   //...but if valid_in or stall_in are very early, we still need the fifo to exit from reset safe state later than usual
        .enable_ecc                     (enable_ecc)
    )
    llram_fifo_inst
    (
        .clock                          (clock),
        .resetn                         (resetn_synchronized),
        
        .valid_in                       (fifo_valid_in_E),
        .data_in                        (data_in),
        .stall_out                      (stall_out),
        .almost_full                    (almost_full),
        
        .valid_out                      (),
        .data_out                       (fifo_data_out),
        .stall_in                       (stall_in_E),
        .almost_empty                   (), //almost_empty is managed externally because read_used_words changes before write_used_words
        .forced_read_out                (),
        
        //special signals exposed by llram fifo needed by zlram fifo
        .zlram_occ_gte_one_E            (occ_gte_one_E),    //this is basically early valid_out from the latency one fifo
        .zlram_stall_out_E              (stall_out_E),      //early stall_out
        
        .ecc_err_status                 (ecc_err_status)
    );
    
    
    
    ////////////////////////////////////////////////////////////////
    //                                                            //
    //  Wrapper logic to transform latency one into latency zero  //
    //                                                            //
    ////////////////////////////////////////////////////////////////
    
    //write into the fifo if the zero latency bypass is not used
    assign fifo_valid_in_E = valid_in_E & (stall_in_E | occ_gte_one_E);
    
    //valid_out = (read_used_words != 0), happens when there is an incoming write or write_used_words != 0
    //stall_out_E is only used to mask valid_out_E from asserting during reset due to valid_in_E
    assign valid_out_E = occ_gte_one_E | (valid_in_E & ~stall_out_E);
    assign forced_read_out_E = valid_out_E & ~stall_in_E;
    
    //retime to no earliness before exporting signals to outside world
    generate
    if (EARLY_MODE == 1) begin : early1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                occ_gte_one <= 1'b0;
                valid_out <= 1'b0;
                forced_read_out <= 1'b0;
                almost_empty <= 1'b1;
            end
            else begin
                occ_gte_one <= occ_gte_one_E;
                valid_out <= valid_out_E;
                forced_read_out <= forced_read_out_E;
                almost_empty <= almost_empty_E;
                if (~sclrn) begin
                    occ_gte_one <= 1'b0;
                    valid_out <= 1'b0;
                    forced_read_out <= 1'b0;
                    almost_empty <= 1'b1;
                end
            end
        end
    end
    else begin : early0
        assign occ_gte_one = occ_gte_one_E;
        assign valid_out = valid_out_E;
        assign forced_read_out = forced_read_out_E;
        assign almost_empty = almost_empty_E;
    end
    endgenerate
    
    
    
    ///////////////////
    //               //
    //  Output data  //
    //               //
    ///////////////////
    
    // take it from the low latency fifo if it has data (this keeps stuff in order), otherwise the fifo is empty so use the data bypass
    // when holding the output data, the HOLD_DATA_OUT_WHEN_EMPTY parameter to the low latency fifo does not work when data is read directly from the bypass
    // because in this case it never got written into the low latency fifo (and writing it causes lots of complications)
    generate
    if (!HOLD_DATA_OUT_WHEN_EMPTY) begin : normal_data_out
        assign data_out = (occ_gte_one) ? fifo_data_out : data_in;
    end
    else begin : held_data_out
        logic data_sel_pri_E, data_sel_sec_E, data_sel_pri, data_sel_sec;
        logic captured_data_clock_en_E, captured_data_clock_en;
        logic [WIDTH-1:0] captured_data_out;
        
        assign data_sel_pri_E = occ_gte_one_E;
        assign data_sel_sec_E = valid_in_E;
        assign captured_data_clock_en_E = valid_out_E & ~stall_in_E;
        
        if (EARLY_MODE == 1) begin : data_out_e1
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) begin
                    data_sel_pri <= (RESET_EVERYTHING) ? '0 : 'x;
                    data_sel_sec <= (RESET_EVERYTHING) ? '0 : 'x;
                    captured_data_clock_en <= (RESET_EVERYTHING) ? '0 : 'x;
                end
                else begin
                    data_sel_pri <= data_sel_pri_E;
                    data_sel_sec <= data_sel_sec_E;
                    captured_data_clock_en <= captured_data_clock_en_E;
                    if (~sclrn && RESET_EVERYTHING) begin
                        data_sel_pri <= 1'b0;
                        data_sel_sec <= 1'b0;
                        captured_data_clock_en <= 1'b0;
                    end
                end
            end
        end
        else begin : data_out_e0
            assign data_sel_pri = data_sel_pri_E;
            assign data_sel_sec = data_sel_sec_E;
            assign captured_data_clock_en = captured_data_clock_en_E;
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
    
    
    
    
    ////////////////////
    //                //
    //  Almost empty  //
    //                //
    ////////////////////
    
    // Almost empty is very tricky because read_used_words changes before write_used_words. Take advantage of this property: read_used_words = write_used_words + write
    // We use two separate occupancy trackers, one detects the original almost empty threhsold (valid if not writing), the other detects an adjust almost empty threshold (valid if writing into fifo).
    // The value we output for almost_empty is simply muxed between these two occupancy trackers. Things get a bit complicated when we factor in all of the parameterizations that hld_fifo could use.
    
    generate
    if (ALMOST_EMPTY_CUTOFF == 0) begin : empty_almost_empty
        assign almost_empty_E = ~valid_out_E;
    end
    else begin : real_almost_empty
    
        logic fifo_in_reset_E;                                  //early version of fifo_in_reset
        logic write_into_fifo_E, try_write_into_fifo_E;         //are we writing or trying to writ into the fifo
        logic read_from_fifo_E, try_read_from_fifo_E;           //are we reading or trying to read from the fifo
        logic not_almost_empty_z_E, not_almost_empty_m1_E;      //only needed because occ trackers export threshold_reached, we want threshold not reached
        logic almost_empty_z_E, almost_empty_m1_E;              //almost empty with original threshold (valid if not writing), almost empty with adjusted threshold (valid if writing)
        logic write_into_fifo_for_almost_empty_E;               //are we actually writing, there are some complications with NEVER_OVERFLOWS and WRITE_AND_READ_DURING_FULL
        
        assign write_into_fifo_E = valid_in_E & ~stall_out_E;
        assign try_write_into_fifo_E = valid_in_E;
        assign read_from_fifo_E = valid_out_E & ~stall_in_E;
        assign try_read_from_fifo_E = ~stall_in_E;
        
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) fifo_in_reset_E <= 1'b1;
            else begin
                fifo_in_reset_E <= 1'b0;
                if (~resetn_delayed[RESET_RELEASE_DELAY-EARLY_MODE]) fifo_in_reset_E <= 1'b1;
            end
        end
        
        //this is the normal almost empty that we would see if not writing into the fifo, read_used_words = write_used_word
        localparam bit ALMOST_EMPTY_Z_GUARD_INCR_RAW = (ALMOST_EMPTY_CUTOFF == INITIAL_OCCUPANCY) ? 1'b1 : 1'b0;
        localparam bit ALMOST_EMPTY_Z_GUARD_DECR_RAW = (ALMOST_EMPTY_CUTOFF == 0) ? 1'b1 : 1'b0;
        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (ALMOST_EMPTY_CUTOFF + 1),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (0),
            .WRITE_AND_READ_DURING_FULL (WRITE_AND_READ_DURING_FULL),
            .WRITE_AND_READ_DURING_EMPTY(1),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        almost_empty_z_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_ALMOST_EMPTY]),
            .incr_no_overflow           (write_into_fifo_E),
            .incr_raw                   ((ALMOST_EMPTY_Z_GUARD_INCR_RAW) ? write_into_fifo_E : try_write_into_fifo_E),
            .decr_no_underflow          (read_from_fifo_E),
            .decr_raw                   ((ALMOST_EMPTY_Z_GUARD_DECR_RAW) ? read_from_fifo_E : try_read_from_fifo_E),
            .threshold_reached          (not_almost_empty_z_E)
        );
        assign almost_empty_z_E = ~not_almost_empty_z_E;
        
        
        //this is the adjusted almost empty that we would see if writing into the fifo, read_used_words = write_used_word+1
        localparam bit ALMOST_EMPTY_M1_GUARD_INCR_RAW = ((ALMOST_EMPTY_CUTOFF-1) == INITIAL_OCCUPANCY) ? 1'b1 : 1'b0;
        localparam bit ALMOST_EMPTY_M1_GUARD_DECR_RAW = ((ALMOST_EMPTY_CUTOFF-1) == 0) ? 1'b1 : 1'b0;
        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (ALMOST_EMPTY_CUTOFF),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (0),
            .WRITE_AND_READ_DURING_FULL (WRITE_AND_READ_DURING_FULL),
            .WRITE_AND_READ_DURING_EMPTY(1),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        almost_empty_m1_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_ALMOST_EMPTY]),
            .incr_no_overflow           (write_into_fifo_E),
            .incr_raw                   ((ALMOST_EMPTY_M1_GUARD_INCR_RAW) ? write_into_fifo_E : try_write_into_fifo_E),
            .decr_no_underflow          (read_from_fifo_E),
            .decr_raw                   ((ALMOST_EMPTY_M1_GUARD_DECR_RAW) ? read_from_fifo_E : try_read_from_fifo_E),
            .threshold_reached          (not_almost_empty_m1_E)
        );
        assign almost_empty_m1_E = ~not_almost_empty_m1_E;
        
        
        //if writing normally should just be: valid_in & ~stall_out, things get a bit complicated when we have never overflows and write and read during full
        //but these special cases are only relevant when the almost empty cutoff is large enough that valid_in is impacted by fifo full (also note that full asserts during reset)
        if ((ALMOST_EMPTY_CUTOFF == DEPTH) && NEVER_OVERFLOWS) begin
            assign write_into_fifo_for_almost_empty_E = valid_in_E & ~fifo_in_reset_E;
        end
        else if ((ALMOST_EMPTY_CUTOFF == DEPTH) && WRITE_AND_READ_DURING_FULL) begin
            assign write_into_fifo_for_almost_empty_E = valid_in_E & ((~fifo_in_reset_E & ~stall_in_E) | ~stall_out_E);
        end
        else begin
            assign write_into_fifo_for_almost_empty_E = valid_in_E & ~stall_out_E;  //this is identical to write_into_fifo_E
        end
        
        assign almost_empty_E = (write_into_fifo_for_almost_empty_E) ? almost_empty_m1_E : almost_empty_z_E;
    end
    endgenerate
    
endmodule

`default_nettype wire
