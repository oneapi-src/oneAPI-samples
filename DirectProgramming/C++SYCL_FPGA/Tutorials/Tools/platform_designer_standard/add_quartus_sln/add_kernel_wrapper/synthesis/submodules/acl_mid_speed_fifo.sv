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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                 //
//  ACL MID SPEED FIFO                                                                                                                                                                                             //
//  Designed and optimized by: Jason Thong                                                                                                                                                                         //
//                                                                                                                                                                                                                 //
//  DESCRIPTION                                                                                                                                                                                                    //
//  ===========                                                                                                                                                                                                    //
//  This fifo was designed as a replacement for scfifo when scfifo is configured in show-ahead mode and with the output data registered. In this configuration the write to read latency is 3 clocks. This fifo    //
//  has both a higher fmax and lower area than scfifo under most configurations (no usage of almost_empty, consume only one of full or almost_full). This is also meant as a replacement for acl_high_speed_fifo.  //
//  Although high speed fifo has a higher fmax, the area is penalty is often not worthwhile if other parts of the circuit limit the fmax to something significantly lower.                                         //
//                                                                                                                                                                                                                 //
//  FUNCTIONAL COMPATIBILITY WITH SCFIFO                                                                                                                                                                           //
//  ====================================                                                                                                                                                                           //
//  There are two major differences in behavior between this fifo and scfifo:                                                                                                                                      //
//                                                                                                                                                                                                                 //
//  1. Reset behavior: During reset this fifo asserts both full and empty, basically it refuses to interact with its surroundings until the reset has propagated through the entire fifo. When using synchronous   //
//     reset not every register is reset, only the critical registers are reset and their reset state propagates to others.                                                                                        //
//                                                                                                                                                                                                                 //
//  2. Almost empty: One of the annoying aspects of scfifo is that for small almost empty thresholds, there are situations where empty=1 and almost_empty=0, which logically makes no sense. The reason for this   //
//     is almost_empty in scfifo is based on write_used_words, the number of words that have been written into the fifo. This is distinct from read_used_words, which is the number of words readable from the     //
//     fifo. They are not the same because it takes time a word written into the fifo to become readable. In this fifo, almost_empty behaves properly since it is based on read_used_words.                        //
//                                                                                                                                                                                                                 //
//  Other differences that can removed by setting certain parameters:                                                                                                                                              //
//                                                                                                                                                                                                                 //
//  3. Read data when empty: This fifo may output X when the fifo is empty, which might actually be useful for simulation to ensure that data is not being consumed when the fifo is empty. Conversely, scfifo     //
//     holds the last data that was read when the fifo is empty. On both fifos, the read data is undefined until the first write. To mimic scfifo behavior with this fifo, set HOLD_DATA_OUT_WHEN_EMPTY=1.         //
//                                                                                                                                                                                                                 //
//  REQUIRED FILES                                                                                                                                                                                                 //
//  ==============                                                                                                                                                                                                 //
//  - acl_mid_speed_fifo.sv                                                                                                                                                                                        //
//  - acl_lfsr.sv                                                                                                                                                                                                  //
//  - acl_tessellated_incr_decr_threshold.sv                                                                                                                                                                       //
//  - acl_reset_handler.sv                                                                                                                                                                                         //
//                                                                                                                                                                                                                 //
//  STALL_IN_EARLINESS AND VALID_IN_EARLINESS                                                                                                                                                                      //
//  =========================================                                                                                                                                                                      //
//  In this fifo earliness is used as an fmax optimization, unlike high speed fifo which used it primarily as an area savings. There are 3 modes of earliness:                                                     //
//                                                                                                                                                                                                                 //
//  1. No earliness: this is the zero-clock-cycle handshaking used by scfifo.                                                                                                                                      //
//                                                                                                                                                                                                                 //
//  2. STALL_IN_EARLINESS >= 1: one extra clock cycle of read latency (stall_in to new value in data_out) allows us to register certain timing sensitive signals. There is a very small area increase from this    //
//  but it is generally well worth the extra 100 MHz or so that results from inserting a few registers in key places.                                                                                              //
//                                                                                                                                                                                                                 //
//  3. STALL_IN_EARLINESS >= 2 && VALID_IN_EARLINESS >= 1: this mode is intended for wide fifos which have high fanout from control signals to anything that affects the data path, e.g. the clock enable for the  //
//  read data register inside the M20K or MLAB. Physical memories are only so wide, so we need multiple physical memories for one wide logical memory, so even things like distributing the same address to all    //
//  of the memories becomes a challenge. Compared to the mode above, this mode retimes all of the control logic one clock earlier, so there is more time to distribute those control signals to the data path.     //
//  For example, instead of 1 control register driving 1024 data registers, if we had 2 clocks to distribute this we could instead have 1 control register driving 32 delayed copies of that control register,     //
//  each of which drives 32 of the data register (still 1024 data registers in total). We let Quartus figure out how much replication is needed in the intermediate stages. Obviously there is an area penalty.    //
//                                                                                                                                                                                                                 //
//  The maximum earliness that this fifo can make use of is STALL_IN_EARLINESS = 2 and VALID_IN_EARLINESS = 1. Excess earliness is absorbed before consumption.                                                    //
//                                                                                                                                                                                                                 //
//  RAM_BLOCK_TYPE                                                                                                                                                                                                 //
//  ==============                                                                                                                                                                                                 //
//  There are different implementations based on whether an M20K or MLAB is used. MLABs have a favorable feature of the read address not needing to be registered inside the memory itself. Although we still      //
//  drive the read address with a register, we now have access to the output of that register. For M20K this register is inside the M20K itself and therefore we have no access to its output. For M20K we have    //
//  our own address register in ALM registers which always stays 1 spot ahead of the read address inside the M20K. Every time we update our own read adddress, the clock enable for the M20K read address is       //
//  asserted so that it captures the old value just before the update of own read address. The consequence of this is we cannot let Quartus decide the RAM implementation, the FIFO needs to choose if the ram     //
//  block type has not been explicitly set to M20K or MLAB by the user.                                                                                                                                            //
//                                                                                                                                                                                                                 //
//  HLD_FIFO FEATURES                                                                                                                                                                                              //
//  =================                                                                                                                                                                                              //
//  This fifo is fully featured against all of the parameters exposed by hld_fifo except for REGISTERED_DATA_OUT_COUNT. Read data from this fifo is always registered, and in the case of M20K it comes from the   //
//  hardened read data register inside the M20K itself. It follows the same reset scheme of asserting full and empty during reset.                                                                                 //
//                                                                                                                                                                                                                 //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

`include "acl_parameter_assert.svh"
`default_nettype none

module acl_mid_speed_fifo #(
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
    parameter     RAM_BLOCK_TYPE = "FIFO_TO_CHOOSE", // "MLAB" | "M20K" | "FIFO_TO_CHOOSE" -> different implementations for MLAB vs M20K, so cannot let quartus decide in case MLAB or M20K is not explicitly specified

    //special configurations for higher fmax / low area
    parameter int STALL_IN_EARLINESS = 0,       // how many clock cycles early is stall_in provided, fifo supports up to 2, setting this any higher results in registers to absorb the excess earliness
    parameter int VALID_IN_EARLINESS = 0,       // how many clock cycles early is valid_in provided, fifo can take advantage of 1 clock or valid_in earliness if stall_in is also at least 2 clocks early
    parameter int STALL_IN_OUTSIDE_REGS = 0,    // number of registers on the stall-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
    parameter int VALID_IN_OUTSIDE_REGS = 0,    // number of registers on the valid-in path external to this module that will delay the propagation of x values on reset (e.g. in hld_fifo)
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

    /////////////////////////////////
    //                             //
    //  Parameter legality checks  //
    //                             //
    /////////////////////////////////

    `ACL_PARAMETER_ASSERT(DEPTH >= 1)

    // ensure thresholds for occupancy are within the legal range
    `ACL_PARAMETER_ASSERT((ALMOST_EMPTY_CUTOFF >= 0) && (ALMOST_EMPTY_CUTOFF <= DEPTH))
    `ACL_PARAMETER_ASSERT((ALMOST_FULL_CUTOFF >= 0) && (ALMOST_FULL_CUTOFF <= DEPTH))
    `ACL_PARAMETER_ASSERT((INITIAL_OCCUPANCY >= 0) && (INITIAL_OCCUPANCY <= DEPTH))

    // width stitching in hld_fifo consumes 1 stage of earliness for both stall and valid
    `ACL_PARAMETER_ASSERT((STALL_IN_OUTSIDE_REGS >= 0) && (STALL_IN_OUTSIDE_REGS <= 1))
    `ACL_PARAMETER_ASSERT((VALID_IN_OUTSIDE_REGS >= 0) && (VALID_IN_OUTSIDE_REGS <= 1))

    // do not allow arbitrarily large amounts of earliness, as this delays the exit from reset "safe state"
    `ACL_PARAMETER_ASSERT((STALL_IN_EARLINESS >= 0) && ((STALL_IN_EARLINESS+STALL_IN_OUTSIDE_REGS) <= 10))
    `ACL_PARAMETER_ASSERT((VALID_IN_EARLINESS >= 0) && ((VALID_IN_EARLINESS+VALID_IN_OUTSIDE_REGS) <= 10))



    //////////////////////////
    //                      //
    //  Parameter settings  //
    //                      //
    //////////////////////////

    // fifo configuration
    localparam int ADDR_RAW             = $clog2(DEPTH);
    localparam int ADDR                 = (ADDR_RAW < 2) ? 2 : ADDR_RAW;                                                            // minimum size of lfsr
    localparam bit USE_MLAB             = (RAM_BLOCK_TYPE == "MLAB") ? 1 : (RAM_BLOCK_TYPE == "M20K") ? 0 : (ADDR <= 5) ? 1 : 0;    //0 = mlab, 1 = m20k

    // earliness configuration
    localparam int EARLY_MODE           = (STALL_IN_EARLINESS >= 2 && VALID_IN_EARLINESS >= 1) ? 2 : (STALL_IN_EARLINESS >= 1) ? 1 : 0;
    localparam int EARLY_VALID          = (EARLY_MODE == 2) ? 1 : 0;
    localparam int EXCESS_EARLY_VALID   = VALID_IN_EARLINESS - EARLY_VALID;
    localparam int EARLY_STALL          = EARLY_MODE;
    localparam int EXCESS_EARLY_STALL   = STALL_IN_EARLINESS - EARLY_STALL;

    // reset timing
    localparam int EXCESS_EARLY_STALL_WITH_EXT = EXCESS_EARLY_STALL + STALL_IN_OUTSIDE_REGS;    //early stall is affected by regs outside this module; account for effect on reset timing
    localparam int EXCESS_EARLY_VALID_WITH_EXT = EXCESS_EARLY_VALID + VALID_IN_OUTSIDE_REGS;    //early valid is affected by regs outisde this module; account for effect on reset timing
    localparam int FLUSH_EARLY_PIPES    = (EXCESS_EARLY_STALL_WITH_EXT > EXCESS_EARLY_VALID_WITH_EXT) ? EXCESS_EARLY_STALL_WITH_EXT : EXCESS_EARLY_VALID_WITH_EXT;  // clocks needs to flush excess earliness pipelines
    localparam int RESET_SYNC_DEPTH     = (SYNCHRONIZE_RESET) ? 3 : 0;                                                   // how many registers are added inside acl_reset_handler for synchronizing the reset
    localparam int RESET_PIPE_DEPTH     = 2;                                                                                    // how many pipeline stages we add to sclrn
    localparam int RESET_LATENCY        = (ASYNC_RESET || RESET_EVERYTHING) ? 0 : (RESET_SYNC_DEPTH + RESET_PIPE_DEPTH);        // how many clocks from the resetn input signal until the reset is consumed
    localparam int MIN_RESET_DELAY      = EARLY_VALID;                                                                          // internal occ tracking is 1 clock early when EARLY_VALID=1, 1 clock to propagate to stall_out
    localparam int RAW_RESET_DELAY      = FLUSH_EARLY_PIPES - RESET_LATENCY;                                                    // delay fifo exit from safe state if need more clocks to flush earliness than reset latency
    localparam int RESET_RELEASE_DELAY  = (RAW_RESET_DELAY < MIN_RESET_DELAY) ? MIN_RESET_DELAY : RAW_RESET_DELAY;              // how many clocks late the fifo exits from safe state

    // reset release delay for the various occupancy trackers
    localparam int RESET_DELAY_ADDR_MATCH   = RESET_RELEASE_DELAY + 1 - EARLY_STALL;
    localparam int RESET_DELAY_STALL_OUT    = RESET_RELEASE_DELAY - EARLY_VALID;
    localparam int RESET_DELAY_ALMOST_FULL  = RESET_RELEASE_DELAY;
    localparam int RESET_DELAY_ALMOST_EMPTY = RESET_RELEASE_DELAY + 2;      // + 2 is actually WRITE_TO_READ_LATENCY - 1
    localparam int RESET_DELAY_MAX          = RESET_DELAY_ALMOST_EMPTY;     // this will always be the largest

    // properties of the fifo which are consumed by the testbench
    localparam int WRITE_TO_READ_LATENCY            = 3;    //once something is written into the fifo, how many clocks later will it be visible on the read side
    localparam int RESET_EXT_HELD_LENGTH            = 4;    //if RESET_EXTERNALLY_HELD = 1, how many clocks does reset need to be held for
    localparam int MAX_CLOCKS_TO_ENTER_SAFE_STATE   = 2;    //upon assertion of reset, worse case number of clocks until fifo shows both full and empty
    localparam int MAX_CLOCKS_TO_EXIT_SAFE_STATE    = 18;   //upon release of reset, worse case number of clocks until fifo is ready to transact (not necessarily observable if INITIAL_OCCUPANCY = DEPTH)



    ///////////////////////////
    //                       //
    //  Signal declarations  //
    //                       //
    ///////////////////////////

    // Naming convention: some signals are retimed early, any signal ending with _ES has the same earliness as the EARLY_STALL parameter. Likewise any signal ending with _EV has the same earliness as EARLY_VALID.

    //reset
    genvar g;
    logic aclrn, sclrn;                                 //these are the typical active low reset signals that are consumed
    logic sclrn_early_two, sclrn_early, sclrn_late;     //helpers for sclrn
    logic [RESET_DELAY_MAX:0] resetn_delayed;           //delayed versions of aclrn or sclrn, consumed by the occupancy trackers
    logic fifo_in_reset;                                //intended primarily for consumption by testbench to know when fifo is in reset, also used for stall_out when NEVER_OVERFLOWS=1

    //absorb excess earliness that the fifo cannot take advantage of
    logic stall_in_ES, valid_in_EV;
    logic [EXCESS_EARLY_STALL:0] stall_in_pipe;
    logic [EXCESS_EARLY_VALID:0] valid_in_pipe;

    //write control
    logic write_into_fifo, write_into_fifo_EV;          //are we writing into the fifo
    logic try_write_into_fifo, try_write_into_fifo_EV;  //are we trying to write into fifo, not necessarily will write, the purpose of this is to shrink the logic cone of threshold_reached in occ trackers
    logic [ADDR-1:0] ram_wr_addr, ram_wr_addr_EV;       //write address to the m20k or mlab
    logic ram_wr_en;                                    //write enable to the m20k or mlab

    //addr match related -- has the write address advanced further than the read address, e.g. is there data in the m20k or mlab that can be consumed
    logic write_into_fifo_ESM1;                         //retimed version of write_into_fifo with the same earliness as EARLY_STALL-1
    logic wr_addr_ahead_of_rd_addr_ES;                  //is write address ahead of read address

    //read control
    logic try_feed_prefetch_ES;                         //try to feed the prefetch if it is empty or fifo is being read
    logic feed_prefetch_ES;                             //actually feed the prefetch it we are trying to and write address is ahead of read address
    logic ram_rd_addr_incr_EV;                          //if feeding prefetch, advance the read pointer...
    logic m20k_addr_b_clock_en, m20k_addr_b_clock_en_EV;//...normally we should also propagate this to the m20k read address, there is a special scenario during reset exit, see comments below
    logic [ADDR-1:0] ram_rd_addr, ram_rd_addr_EV;       //read address to the m20k or mlab

    //prefetch control
    logic prefetch_clock_en;                            //clock enable for the prefetch, which is actually the hardened output read data registers inside the m20k
    logic valid_out_ES;                                 //early valid_out

    //occupancy trackers -- stall_out, almost_full, almost_empty
    logic stall_out_EV;                                 //early stall_out
    logic read_from_fifo, read_from_fifo_ES, read_from_fifo_EV;                 //are we reading from the fifo
    logic try_read_from_fifo, try_read_from_fifo_ES, try_read_from_fifo_EV;     //are we trying to read from fifo, not necessarily will read, purpose is to simplify threshold_reached logic in occ trackers



    /////////////
    //         //
    //  Reset  //
    //         //
    /////////////

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
        .o_resetn_synchronized  (),
        .o_sclrn                (sclrn_early_two)
    );

    generate
    if (ASYNC_RESET) begin : async_reset
        assign sclrn = 1'b1;
        assign sclrn_late = 1'b1;
        assign sclrn_early = 1'b1;
    end
    else begin : sync_reset
        logic [2:0] sclrn_chain;    //pulse extend from 1 clock to 4 clocks
        always_ff @(posedge clock) begin
            sclrn_chain <= (sclrn_chain << 1) | sclrn_early_two;
            sclrn_early <= (RESET_EXTERNALLY_HELD) ? sclrn_early_two : ((&sclrn_chain) & sclrn_early_two);
            sclrn <= sclrn_early;
            sclrn_late <= sclrn;    //only the read address increment consumes this when using m20k and sclr
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

    //this signal is consumed by the testbench to know whether the fifo is still in reset, can't use stall_out when fifo starts as full
    //this signal may be exported by the fifo in certain configurations, e.g. NEVER_OVERFLOWS=1
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
    assign stall_in_ES = stall_in_pipe[EXCESS_EARLY_STALL];
    assign valid_in_EV = valid_in_pipe[EXCESS_EARLY_VALID];



    ////////////////////
    //                //
    //  Memory block  //
    //                //
    ////////////////////

    // Usage of altdpram - unlike the M20K in which it is impossible to bypass the input registers (addresses, write data, write enable), for the MLAB it is possible to bypass the input
    // register for the read address. There is no parameterization of altera_syncram that supports this, hence the use of altdpram.

    // It is desirable to have access to the output of read address address. In the case of MLAB, the read address is driven by ALM registers. For M20K, we have no visibility on the output
    // of the read address register because this is a hardened register inside the M20K itself. For M20K we have our own read address in ALM registers which is always 1 ahead of the hardened
    // read address inside the M20K. Only when we update our read address, we assert the clock enable for the hardened read address inside the M20K, this way it always captures 1 value behind
    // what our ALM register read address is. For asynchronous reset, the M20K read address is aclr to 0 and our address starts at 1 (actually INITIAL_OCCPUPANCY=1 on the LFSR). For synchronous
    // reset, the M20K read address clock enable is active during reset so that we can clock in the value of our ALM register read address, upon reset exit the M20K clock enable is shut off
    // and our read address advances 1 step forward.

    assign ecc_err_status = 2'h0;   // ECC IS NOT IMPLEMENTED YET, see case:555783

    assign ram_wr_en = write_into_fifo;

    generate
    if (WIDTH > 0) begin : gen_ram
        if (USE_MLAB) begin : gen_mlab
            altdpram #(     //modelsim library: altera_mf
                .indata_aclr ("OFF"),
                .indata_reg ("INCLOCK"),
                .intended_device_family ("Stratix 10"),//quartus will correct this automatically to whatever your project actually uses
                .lpm_type ("altdpram"),
                .ram_block_type ("MLAB"),
                .outdata_aclr ("OFF"),
                .outdata_sclr ("OFF"),
                .outdata_reg ("OUTCLOCK"),          //output data is registered, clock enable for this is controlled by outclocken
                .rdaddress_aclr ("OFF"),
                .rdaddress_reg ("UNREGISTERED"),    //we own the read address, bypass the equivalent of the internal address_b from m20k
                .rdcontrol_aclr ("OFF"),
                .rdcontrol_reg ("UNREGISTERED"),
                .read_during_write_mode_mixed_ports ("DONT_CARE"),
                .width (WIDTH),
                .widthad (ADDR),
                .width_byteena (1),
                .wraddress_aclr ("OFF"),
                .wraddress_reg ("INCLOCK"),
                .wrcontrol_aclr ("OFF"),
                .wrcontrol_reg ("INCLOCK")
            )
            altdpram_component
            (
                //clock, no reset
                .inclock    (clock),
                .outclock   (clock),

                //write port
                .data       (data_in),
                .wren       (ram_wr_en),
                .wraddress  (ram_wr_addr),

                //read port
                .rdaddress  (ram_rd_addr),
                .outclocken (prefetch_clock_en),
                .q          (data_out),

                //unused
                .aclr (1'b0),
                .sclr (1'b0),
                .byteena (1'b1),
                .inclocken (1'b1),
                .rdaddressstall (1'b0),
                .rden (1'b1),
                .wraddressstall (1'b0)
            );
        end
        else begin : gen_m20k
            altera_syncram #(   //modelsim library: altera_lnsim
                .numwords_a (2**ADDR),
                .numwords_b (2**ADDR),
                .address_aclr_b ((ASYNC_RESET) ? "CLEAR1" : "NONE"),
                .address_reg_b ("CLOCK1"),
                .clock_enable_input_a ("BYPASS"),
                .clock_enable_input_b ("BYPASS"),
                .clock_enable_output_b ("NORMAL"),      //clock enable for output data register is controlled by clocken1
                .enable_ecc ("FALSE"),
                .intended_device_family ("Stratix 10"), //quartus will correct this automatically to whatever your project actually uses
                .lpm_type ("altera_syncram"),
                .operation_mode ("DUAL_PORT"),
                .outdata_aclr_b ("NONE"),
                .outdata_sclr_b ("NONE"),
                .outdata_reg_b ("CLOCK1"),              //output data is registered, clock1 and clock0 come from the same source, using clock1 so that we gain acceses to certain clock enables
                .power_up_uninitialized ("TRUE"),
                .ram_block_type ("M20K"),
                .read_during_write_mode_mixed_ports ("DONT_CARE"),
                .widthad_a (ADDR),
                .widthad_b (ADDR),
                .width_a (WIDTH),
                .width_b (WIDTH),
                .width_byteena_a (1)
            )
            altera_syncram
            (
                //clock and reset
                .clock0         (clock),
                .clock1         (clock),
                .aclr1          ((ASYNC_RESET) ? ~aclrn : 1'b0),    //this is used to reset the internal address_b when ASYNC_RESET=1

                //write port
                .wren_a         (ram_wr_en),
                .address_a      (ram_wr_addr),
                .data_a         (data_in),

                //read port
                .address_b      (ram_rd_addr),
                .addressstall_b (~m20k_addr_b_clock_en),
                .clocken1       (prefetch_clock_en),
                .q_b            (data_out),

                //unused
                .aclr0 (1'b0),
                .address2_a (1'b1),
                .address2_b (1'b1),
                .addressstall_a (1'b0),
                .byteena_a (1'b1),
                .byteena_b (1'b1),
                .clocken0 (1'b1),
                .clocken2 (1'b1),
                .clocken3 (1'b1),
                .data_b ({WIDTH{1'b1}}),
                .eccencbypass (1'b0),
                .eccencparity (8'b0),
                .eccstatus (),
                .q_a (),
                .rden_a (1'b1),
                .rden_b (1'b1),
                .sclr (1'b0),
                .wren_b (1'b0)
            );
        end
    end
    endgenerate



    ///////////////////////////////////////////////
    //                                           //
    //  Write logic -- resolve earliness timing  //
    //                                           //
    ///////////////////////////////////////////////

    // all of the write logic is retimed early by EARLY_VALID, we need to restore no earliness timing for the ram and some occupancy trackers (almost_full, almost_empty)

    assign try_write_into_fifo_EV = valid_in_EV;
    assign write_into_fifo_EV = valid_in_EV & ~stall_out_EV;

    generate
    if (EARLY_VALID == 0) begin : write_ev0
        assign write_into_fifo = write_into_fifo_EV;
        assign try_write_into_fifo = try_write_into_fifo_EV;
        assign ram_wr_addr = ram_wr_addr_EV;
    end
    if (EARLY_VALID == 1) begin : write_ev1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                write_into_fifo <= 1'b0;
                ram_wr_addr <= '0;
            end
            else begin
                write_into_fifo <= write_into_fifo_EV;
                ram_wr_addr <= ram_wr_addr_EV;
                if (~sclrn) write_into_fifo <= 1'b0;
                if (~sclrn && RESET_EVERYTHING) ram_wr_addr <= '0;
            end
        end
        assign try_write_into_fifo = write_into_fifo;
    end
    endgenerate



    /////////////////////
    //                 //
    //  Write address  //
    //                 //
    /////////////////////

    acl_lfsr #(
        .WIDTH                  (ADDR),
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_RESET      (0),
        .INITIAL_OCCUPANCY      (INITIAL_OCCUPANCY)
    )
    ram_wr_addr_inst
    (
        .clock                  (clock),
        .resetn                 (resetn_delayed[0]),
        .enable                 (write_into_fifo_EV),
        .state                  (ram_wr_addr_EV)
    );



    //////////////////////////////////////////////
    //                                          //
    //  Read logic -- resolve earliness timing  //
    //                                          //
    //////////////////////////////////////////////

    // The DECISION on whether to read from the ram is retimed early by EARLY_STALL clock cycles. The UPDATE of the read address itself is retimed early by EARLY_VALID clock cycles.
    // When EARLY_STALL is 0, then EARLY_VALID == EARLY_STALL. When EARLY_STALL is 1 or larger, then EARLY_VALID == EARLY_STALL-1, so in these cases the decision to read from the ram
    // happens 1 clock cycle before read address itself is updated. This allows the decision to be registered first which helps fmax. In the case of EARLY_VALID = 1, there is an extra
    // pipeline stage from the read address to the M20K or MLAB.

    // Decision on whether to read from the ram: does the prefetch have space right now (~valid_out_ES) or will it have space because we are reading (~stall_in_ES) -> this means we COULD feed the prefetch.
    // In order to actually feed the prefetch, we also need the write address to be ahead of the read address.
    assign try_feed_prefetch_ES = ~valid_out_ES | ~stall_in_ES;
    assign feed_prefetch_ES = wr_addr_ahead_of_rd_addr_ES & try_feed_prefetch_ES;

    // Normally we should advance the read address when we are able to feed the prefetch.
    // When using M20K we don't have access to the address_b register inside the M20K, so we run our own read address register 1 spot ahead, and when it updates then only we clock it into the M20K.
    // For aclr, we reset the address_b register inside the M20K, and ram_rd_addr resets to 1 ahead.
    // For sclr, during reset ram_rd_addr is 0 and that gets clocked into the M20K, upon exit of reset we no longer clock ram_rd_addr into the M20K and ram_rd_addr moves ahead by 1.
    // Note the update of the read address itself is based on EARLY_VALID whereas the decision to read is based on EARLY_STALL, so we have to convert earliness settings.
    generate
    if (EARLY_STALL == 0) begin : read_incr0    //EARLY_VALID == EARLY_STALL
        assign ram_rd_addr_incr_EV     = (!USE_MLAB && !ASYNC_RESET) ? (feed_prefetch_ES | ~sclrn_late) : feed_prefetch_ES;
        assign m20k_addr_b_clock_en_EV = (!USE_MLAB && !ASYNC_RESET) ? (feed_prefetch_ES | ~sclrn)      : feed_prefetch_ES;
    end
    if (EARLY_STALL >= 1) begin : read_incr12   //EARLY_VALID == EARLY_STALL-1, one pipeline stage needed for EV to consume from ES
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                ram_rd_addr_incr_EV <= 1'b0;
                m20k_addr_b_clock_en_EV <= 1'b0;
            end
            else begin
                ram_rd_addr_incr_EV <= feed_prefetch_ES;
                m20k_addr_b_clock_en_EV <= feed_prefetch_ES;
                if (!USE_MLAB && !ASYNC_RESET) begin    //special reset behavior for M20K using sync reset, peek one stage ahead on sclr since we are now registering these signals
                    if (~sclrn) ram_rd_addr_incr_EV <= 1'b1;
                    if (~sclrn_early) m20k_addr_b_clock_en_EV <= 1'b1;
                end
                else begin                              //normal reset behavior
                    if (~sclrn && RESET_EVERYTHING) begin
                        ram_rd_addr_incr_EV <= 1'b0;
                        m20k_addr_b_clock_en_EV <= 1'b0;
                    end
                end
            end
        end
    end
    endgenerate



    ////////////////////
    //                //
    //  Read address  //
    //                //
    ////////////////////

    acl_lfsr #(
        .WIDTH                  (ADDR),
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_RESET      (0),
        .INITIAL_OCCUPANCY      ((!USE_MLAB && ASYNC_RESET) ? 1 : 0)    //for M20K our read address is 1 ahead of the address_b inside the M20K, for async reset we don't clock in our address during reset, so just start 1 ahead
    )
    ram_rd_addr_inst
    (
        .clock                  (clock),
        .resetn                 (resetn_delayed[0]),
        .enable                 (ram_rd_addr_incr_EV),
        .state                  (ram_rd_addr_EV)
    );

    //correct the timing for the signals driving the ram itself
    generate
    if (EARLY_VALID == 0) begin : read_addr_ev0
        assign ram_rd_addr = ram_rd_addr_EV;
        assign m20k_addr_b_clock_en = m20k_addr_b_clock_en_EV;
    end
    if (EARLY_VALID == 1) begin : read_addr_ev1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                ram_rd_addr <= '0;
                m20k_addr_b_clock_en <= 1'b0;
            end
            else begin
                ram_rd_addr <= ram_rd_addr_EV;
                m20k_addr_b_clock_en <= m20k_addr_b_clock_en_EV;
                if (~sclrn && RESET_EVERYTHING) begin
                    ram_rd_addr <= '0;
                    m20k_addr_b_clock_en <= (!USE_MLAB && !ASYNC_RESET) ? 1'b1 : 1'b0;  //special reset behavior for M20K using sync reset
                end
            end
        end
    end
    endgenerate



    /////////////////////////////////////
    //                                 //
    //  Address match lookahead logic  //
    //                                 //
    /////////////////////////////////////

    // The memory block is configured so that reading and writing from the same address returns X. A read is valid once the write address has advanced past the read address. In tracking the number
    // of "readable" addresses, we therefore increment one clock cycle after the write has happened. We decrement as soon as the read happens.

    // Whether or not any addresses are "readable" has been retimed by EARLY_STALL clocks ahead. The read side logic has all be retimed early by EARLY_STALL clocks. The write side logic is retimed
    // ahead by EARLY_VALID clocks, so we need to do an earliness convesion. When EARLY_STALL = 0, the increment should happen 1 clock after the write. When EARLY_STALL = 1, we still have EARLY_VALID = 0
    // so the increment should happen at the same time as the write. When EARLY_STALL = 2 we have EARLY_VALID = 1, so the increment should happen at the same time as the write.

    generate
    if (EARLY_STALL == 0) begin : addr_match_es0    //EARLY_VALID == EARLY_STALL  -> increment should happen 1 clock after the write into the ram
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) write_into_fifo_ESM1 <= 1'b0;
            else begin
                write_into_fifo_ESM1 <= write_into_fifo_EV;
                if (~sclrn) write_into_fifo_ESM1 <= 1'b0;
            end
        end
    end
    if (EARLY_STALL >= 1) begin : addr_match_es12   //EARLY_VALID == EARLY_STALL-1  -> increment should happen on the same clock as the write into the ram
        assign write_into_fifo_ESM1 = write_into_fifo_EV;
    end
    endgenerate

    acl_tessellated_incr_decr_threshold #(
        .CAPACITY                   (DEPTH),
        .THRESHOLD                  (1),
        .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
        .THRESHOLD_REACHED_AT_RESET (0),
        .WRITE_AND_READ_DURING_FULL (0),
        .ASYNC_RESET                (ASYNC_RESET),
        .SYNCHRONIZE_RESET          (0),
        .RESET_EVERYTHING           (RESET_EVERYTHING),
        .RESET_EXTERNALLY_HELD      (1)
    )
    addr_match_inst
    (
        .clock                      (clock),
        .resetn                     (resetn_delayed[RESET_DELAY_ADDR_MATCH]),
        .incr_no_overflow           (write_into_fifo_ESM1),
        .incr_raw                   (write_into_fifo_ESM1), //do not use unguarded incr, with initial occupancy = 0 and valid_in = 1 during reset, threshold_reached will assert 1 clock too soon coming out of reset
        .decr_no_underflow          (feed_prefetch_ES),
        .decr_raw                   (try_feed_prefetch_ES),
        .threshold_reached          (wr_addr_ahead_of_rd_addr_ES)
    );



    //////////////////
    //              //
    //  Fifo empty  //
    //              //
    //////////////////

    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) valid_out_ES <= 1'b0;
        else begin
            if (wr_addr_ahead_of_rd_addr_ES & ~valid_out_ES) valid_out_ES <= 1'b1;
            if (~wr_addr_ahead_of_rd_addr_ES & ~stall_in_ES) valid_out_ES <= 1'b0;
            if (~sclrn) valid_out_ES <= 1'b0;
        end
    end



    //////////////////////////////////////
    //                                  //
    //  Read valid retiming resolution  //
    //                                  //
    //////////////////////////////////////

    // Read side logic is computed early, restore no earliness timing of signals for occupancy trackers, valid_out, and prefetch clock enable

    assign read_from_fifo_ES = valid_out_ES & ~stall_in_ES;     //will we FORCE a read from the fifo in EARLY_STALL clocks from now
    assign try_read_from_fifo_ES = ~stall_in_ES;                //could we TRY to read from the fifo in EARLY_STALL clocks from now, e.g. does downstream have space for more data
    assign forced_read_out = read_from_fifo;                    //are we FORCING a read from the fifo right now

    generate
    if (EARLY_STALL == 0) begin : read_es0
        assign valid_out = valid_out_ES;
        assign prefetch_clock_en = (!HOLD_DATA_OUT_WHEN_EMPTY) ? try_feed_prefetch_ES : feed_prefetch_ES;
        assign read_from_fifo_EV = read_from_fifo_ES;
        assign try_read_from_fifo_EV = try_read_from_fifo_ES;
        assign read_from_fifo = read_from_fifo_ES;
        assign try_read_from_fifo = try_read_from_fifo_ES;
    end
    if (EARLY_STALL == 1) begin : read_es1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                valid_out <= 1'b0;
                prefetch_clock_en <= (!HOLD_DATA_OUT_WHEN_EMPTY) ? 1'b1 : 1'b0;
                read_from_fifo_EV <= 1'b0;
            end
            else begin
                valid_out <= valid_out_ES;
                prefetch_clock_en <= (!HOLD_DATA_OUT_WHEN_EMPTY) ? try_feed_prefetch_ES : feed_prefetch_ES;
                read_from_fifo_EV <= read_from_fifo_ES;
                if (~sclrn) begin
                    valid_out <= 1'b0;
                    read_from_fifo_EV <= 1'b0;
                end
                if (~sclrn && RESET_EVERYTHING) prefetch_clock_en <= (!HOLD_DATA_OUT_WHEN_EMPTY) ? 1'b1 : 1'b0;
            end
        end
        assign try_read_from_fifo_EV = read_from_fifo_EV;
        assign read_from_fifo = read_from_fifo_EV;
        assign try_read_from_fifo = read_from_fifo_EV;
    end
    if (EARLY_STALL == 2) begin : read_es2
        logic valid_out_early, prefetch_clock_en_early;
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                valid_out_early <= 1'b0;
                valid_out <= 1'b0;
                prefetch_clock_en_early <= (!HOLD_DATA_OUT_WHEN_EMPTY) ? 1'b1 : 1'b0;
                prefetch_clock_en <= (!HOLD_DATA_OUT_WHEN_EMPTY) ? 1'b1 : 1'b0;
                read_from_fifo_EV <= 1'b0;
                read_from_fifo <= 1'b0;
            end
            else begin
                valid_out_early <= valid_out_ES;
                valid_out <= valid_out_early;
                prefetch_clock_en_early <= (!HOLD_DATA_OUT_WHEN_EMPTY) ? try_feed_prefetch_ES : feed_prefetch_ES;
                prefetch_clock_en <= prefetch_clock_en_early;
                read_from_fifo_EV <= read_from_fifo_ES;
                read_from_fifo <= read_from_fifo_EV;
                if (~sclrn) begin
                    valid_out <= 1'b0;
                    read_from_fifo_EV <= 1'b0;
                end
                if (~sclrn && RESET_EVERYTHING) begin
                    valid_out_early <= 1'b0;
                    prefetch_clock_en_early <= (!HOLD_DATA_OUT_WHEN_EMPTY) ? 1'b1 : 1'b0;
                    prefetch_clock_en <= (!HOLD_DATA_OUT_WHEN_EMPTY) ? 1'b1 : 1'b0;
                    read_from_fifo <= 1'b0;
                end
            end
        end
        assign try_read_from_fifo_EV = read_from_fifo_EV;
        assign try_read_from_fifo = read_from_fifo;
    end
    endgenerate



    /////////////////
    //             //
    //  Fifo full  //
    //             //
    /////////////////

    // Suppose we are trying to detect a threshold of N. We reach the threshold when the occupancy is N-1 and we are increasing, and we have no longer reached the threshold
    // when the occupancy is N and we are decreasing. The occupancy has to be tracked using the guarded increment and decrement (incr_no_overflow and decr_no_underflow),
    // but the increasing or decreasing used to change the state of threshold_reached does not need to be guarded if the threshold N is far away from overflow or underflow.
    //
    // This fifo has a 3 clock latency from write to read, so it is possible that write_used_words can be at fifo capacity (fifo is full) while stall_in is off (downstream wants
    // to read) and valid_out is off (no data to provide to downstream). For a threshold of 3 or lower, we are not sufficiently far from underflow to use an unguarded decrement.
    //
    // Likewise if the initial occupancy is close to the threshold, unguarded increment and decrement can cause similar problems. However in the case of stall_out, valid_in=1
    // during reset will simply be masked by reset. Once stall_out deasserts, we have to guard against decr_raw until valid_out asserts from initial occupancy, there is a gap of
    // 2 clocks here (write to read latency minus 1). If the initial occupancy is within 2 of the capacity (DEPTH), we must use decr_no_underflow instead.

    generate
    if (NEVER_OVERFLOWS) begin : gen_reset_stall_out    //no overflow protection, but upstream still needs a way to know when fifo has exited from reset
        if (EARLY_VALID == 1) begin
            always_ff @(posedge clock or negedge aclrn) begin
                if (~aclrn) stall_out_EV <= 1'b1;
                else begin
                    stall_out_EV <= 1'b0;
                    if (~resetn_delayed[RESET_RELEASE_DELAY-1]) stall_out_EV <= 1'b1;   //RESET_RELEASE_DELAY will be at least 2 when EARLY_VALID is 1
                end
            end
        end
        else begin
            assign stall_out_EV = fifo_in_reset;
        end
    end
    else begin : gen_real_stall_out

        localparam bit STALL_OUT_GUARD_DECR_RAW = ((DEPTH <= WRITE_TO_READ_LATENCY) || ((DEPTH-INITIAL_OCCUPANCY) <= (WRITE_TO_READ_LATENCY-1))) ? 1'b1 : 1'b0;
        logic stall_out_EV_raw;

        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (DEPTH),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (1),
            .WRITE_AND_READ_DURING_FULL (WRITE_AND_READ_DURING_FULL),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        stall_out_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_STALL_OUT]),
            .incr_no_overflow           (write_into_fifo_EV),
            .incr_raw                   (try_write_into_fifo_EV),
            .decr_no_underflow          (read_from_fifo_EV),
            .decr_raw                   ((STALL_OUT_GUARD_DECR_RAW) ? read_from_fifo_EV : try_read_from_fifo_EV),
            .threshold_reached          (stall_out_EV_raw)
        );

        assign stall_out_EV = (!WRITE_AND_READ_DURING_FULL) ? stall_out_EV_raw : (stall_out_EV_raw & ~read_from_fifo_EV);

    end
    endgenerate

    generate
    if (EARLY_VALID==0) begin : stall_out0
        assign stall_out = stall_out_EV;
    end
    if (EARLY_VALID==1) begin : stall_out1
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) stall_out <= 1'b1;
            else begin
                stall_out <= stall_out_EV;
                if (~sclrn) stall_out <= 1'b1;
            end
        end
    end
    endgenerate



    ///////////////////
    //               //
    //  Almost full  //
    //               //
    ///////////////////

    generate
    if ((ALMOST_FULL_CUTOFF == 0) && (NEVER_OVERFLOWS == 0) && (WRITE_AND_READ_DURING_FULL == 0)) begin : full_almost_full
        assign almost_full = stall_out;
    end
    else begin : real_almost_full
        //similar idea of read-side protection as stall_out
        localparam int ALMOST_FULL_INIT_OCC_DIFF = ((DEPTH-ALMOST_FULL_CUTOFF) > INITIAL_OCCUPANCY) ? (DEPTH-ALMOST_FULL_CUTOFF-INITIAL_OCCUPANCY) : INITIAL_OCCUPANCY-(DEPTH-ALMOST_FULL_CUTOFF);
        localparam bit ALMOST_FULL_GUARD_DECR_RAW = (((DEPTH-ALMOST_FULL_CUTOFF) <= WRITE_TO_READ_LATENCY) || (ALMOST_FULL_INIT_OCC_DIFF <= (WRITE_TO_READ_LATENCY-1))) ? 1'b1 : 1'b0;

        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (DEPTH - ALMOST_FULL_CUTOFF),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (1),
            .WRITE_AND_READ_DURING_FULL (WRITE_AND_READ_DURING_FULL),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        almost_full_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_ALMOST_FULL]),
            .incr_no_overflow           (write_into_fifo),
            .incr_raw                   (try_write_into_fifo),
            .decr_no_underflow          (read_from_fifo),
            .decr_raw                   ((ALMOST_FULL_GUARD_DECR_RAW) ? read_from_fifo : try_read_from_fifo),
            .threshold_reached          (almost_full)
        );
    end
    endgenerate



    ////////////////////
    //                //
    //  Almost empty  //
    //                //
    ////////////////////

    generate
    if (ALMOST_EMPTY_CUTOFF == 0) begin : empty_almost_empty
        assign almost_empty = ~valid_out;
    end
    else begin : real_almost_empty
        logic not_almost_empty;
        logic write_into_fifo_late, write_into_fifo_late_two;
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                write_into_fifo_late <= 1'b0;
                write_into_fifo_late_two <= 1'b0;
            end
            else begin
                write_into_fifo_late <= write_into_fifo;
                write_into_fifo_late_two <= write_into_fifo_late;
                if (~sclrn && RESET_EVERYTHING) begin
                    write_into_fifo_late <= 1'b0;
                    write_into_fifo_late_two <= 1'b0;
                end
            end
        end

        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DEPTH),
            .THRESHOLD                  (ALMOST_EMPTY_CUTOFF + 1),
            .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
            .THRESHOLD_REACHED_AT_RESET (0),
            .WRITE_AND_READ_DURING_FULL (0),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0),
            .RESET_EVERYTHING           (RESET_EVERYTHING),
            .RESET_EXTERNALLY_HELD      (1)
        )
        almost_empty_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_delayed[RESET_DELAY_ALMOST_EMPTY]),
            .incr_no_overflow           (write_into_fifo_late_two),
            .incr_raw                   (write_into_fifo_late_two),
            .decr_no_underflow          (read_from_fifo),
            .decr_raw                   (try_read_from_fifo),
            .threshold_reached          (not_almost_empty)
        );
        assign almost_empty = ~not_almost_empty;
    end
    endgenerate

endmodule

`default_nettype wire
