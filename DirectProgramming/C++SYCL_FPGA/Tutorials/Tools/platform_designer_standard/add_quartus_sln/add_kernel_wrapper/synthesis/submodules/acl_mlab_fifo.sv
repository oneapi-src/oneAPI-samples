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


// This is a latency 2 MLAB only variant of acl_mid_speed_fifo. Use hld_fifo with STYLE = "mlab" to gain access to this fifo implementation.
//
// Starting from empty, mid speed fifo takes 3 clocks to become non-empty after accepting input data:
// First clock cycle: fifo input data is written to the RAM
// Second clock cycle: fifo issues a read to the address previously written to
// Third clock cycle: read data is available from the RAM, which drives the fifo output data
//
// For RAM blocks that natively support new data mode in mixed port read during write, the fifo can issue the read one clock cycle earlier and get the newly written data.
// MLAB natively supports this mode whereas M20K does not. If one were to build bypass logic around the M20K, it would end up looking similar to acl_latency_one_ram_fifo.
//
// This fifo can benefit from STALL_IN_EARLINESS up to 1, in which there is some manual retiming of fmax sensitive signals. Additional earliness is simply
// absorbed with a shift register, likewise for any VALID_IN_EARLINESS.

`include "acl_parameter_assert.svh"
`default_nettype none

module acl_mlab_fifo #(
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

    //special configurations for higher fmax / low area
    parameter int STALL_IN_EARLINESS = 0,       // how many clock cycles early is stall_in provided, fifo supports up to 1, setting this any higher results in registers to absorb the excess earliness
    parameter int VALID_IN_EARLINESS = 0,       // how many clock cycles early is valid_in provided, fifo does not take advantage of this (all absorbed as excess earliness)
    parameter bit NEVER_OVERFLOWS = 0,          // set to 1 to disable fifo's internal overflow protection, area savings by removing one incr/decr/thresh, stall_out still asserts during reset but won't mask valid_in

    //special features that typically have an fmax penalty
    parameter bit HOLD_DATA_OUT_WHEN_EMPTY = 0, // 0 means data_out can be x when fifo is empty, 1 means data_out will hold last value when fifo is empty (scfifo behavior, has fmax penalty)
    parameter bit WRITE_AND_READ_DURING_FULL = 0,//set to 1 to allow writing and reading while the fifo is full, this may have an fmax penalty, to compensate it is recommended to use this with NEVER_OVERFLOWS = 1

    //error correction code
    parameter enable_ecc = "FALSE"              // NOT IMPLEMENTED YET, see case:555783
)
(
    input  wire                 clock,
    input  wire                 resetn,

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
    //  Parameter legality checks  //
    /////////////////////////////////

    `ACL_PARAMETER_ASSERT(DEPTH >= 1)

    // ensure thresholds for occupancy are within the legal range
    `ACL_PARAMETER_ASSERT((ALMOST_EMPTY_CUTOFF >= 0) && (ALMOST_EMPTY_CUTOFF <= DEPTH))
    `ACL_PARAMETER_ASSERT((ALMOST_FULL_CUTOFF >= 0) && (ALMOST_FULL_CUTOFF <= DEPTH))
    `ACL_PARAMETER_ASSERT((INITIAL_OCCUPANCY >= 0) && (INITIAL_OCCUPANCY <= DEPTH))

    // do not allow arbitrarily large amounts of earliness, as this delays the exit from reset "safe state"
    `ACL_PARAMETER_ASSERT((STALL_IN_EARLINESS >= 0) && (STALL_IN_EARLINESS <= 10))
    `ACL_PARAMETER_ASSERT((VALID_IN_EARLINESS >= 0) && (VALID_IN_EARLINESS <= 10))



    //////////////////////////
    //  Derived parameters  //
    //////////////////////////

    //fifo configuration
    localparam int ADDR                     = (DEPTH <= 4) ? 2 : $clog2(DEPTH);         //address bus width must be at least the minimum size of the lfsr
    localparam int EARLY_STALL              = (STALL_IN_EARLINESS >= 1) ? 1 : 0;        //maximum amount of usable stall in earliness is 1

    //reset timing
    localparam int RESET_SYNC_DEPTH         = (SYNCHRONIZE_RESET) ? 3 : 0;              //latency to synchronize a reset before consumption, specifically the latency to exit from reset
    localparam int RESET_PIPE_DEPTH         = (ASYNC_RESET) ? 0 : 2;                    //how many pipeline stages we add to sclrn after the reset handler
    localparam int RESET_LATENCY            = RESET_SYNC_DEPTH + RESET_PIPE_DEPTH;      //latency from resetn input signal until reset is consumed

    //how many clock cycles late to exit from reset to ensure excess earliness pipelines have been flushed
    localparam int EXCESS_EARLY_STALL_PIPE  = STALL_IN_EARLINESS - EARLY_STALL;         //number of pipeline stages needed to absorb excess stall in earliness
    localparam int EXCESS_EARLY_VALID_PIPE  = VALID_IN_EARLINESS;                       //likewise for valid in earliness, note that valid in earliness is not usable by this fifo
    localparam int EXCESS_EARLY_PIPE        = (EXCESS_EARLY_STALL_PIPE > EXCESS_EARLY_VALID_PIPE) ? EXCESS_EARLY_STALL_PIPE : EXCESS_EARLY_VALID_PIPE;
    localparam int EXCESS_EARLY_RESET       = (ASYNC_RESET || RESET_EVERYTHING) ? RESET_LATENCY : 0;    //if the pipeline is reset, need to account for reset latency as well
    localparam int EXCESS_EARLY_LATENCY     = EXCESS_EARLY_PIPE + EXCESS_EARLY_RESET;   //how many clock cycles late to exit from reset

    //reset latency balancing
    localparam int RESET_DELAY_RAW          = (EXCESS_EARLY_LATENCY > RESET_LATENCY) ? EXCESS_EARLY_LATENCY - RESET_LATENCY : 0;    //delay exiting from reset if more clocks are needed to flush excess earliness
                                                                                                                                    //pipelines compared to the natural latency of the reset pipelining
    localparam int RESET_DELAY              = RESET_DELAY_RAW + EARLY_STALL;            //retiming fmax sensitive signals one clock earlier requires peeking up the reset pipeline, add 1 to allow this
    localparam int RESET_DELAY_ADDR_COMPARE = RESET_DELAY - EARLY_STALL;                //if retiming valid_out 1 clock earlier, internal initial occupancy must be observable one clock earlier
    localparam int RESET_DELAY_WRITE_USEDW  = RESET_DELAY;                              //for anything tracking write used words e.g. almost_full
    localparam int RESET_DELAY_READ_USEDW   = RESET_DELAY + 1;                          //for anything tracking read used words e.g. almost_empty, the +1 is actually +WRITE_TO_READ_LATENCY-1
    localparam int RESET_DELAY_MAX          = RESET_DELAY_READ_USEDW;                   //this will always be the largest

    //properties of the fifo which are consumed by the testbench
    localparam int WRITE_TO_READ_LATENCY            = 2;    //once something is written into the fifo, how many clocks later will it be visible on the read side
    localparam int RESET_EXT_HELD_LENGTH            = 4;    //if RESET_EXTERNALLY_HELD = 1, how many clocks does reset need to be held for
    localparam int MAX_CLOCKS_TO_ENTER_SAFE_STATE   = 3;    //upon assertion of reset, worse case number of clocks until fifo shows both full and empty
    localparam int MAX_CLOCKS_TO_EXIT_SAFE_STATE    = 19;   //upon release of reset, worse case number of clocks until fifo is ready to transact (not necessarily observable if INITIAL_OCCUPANCY = DEPTH)



    ///////////////////////////
    //  Signal declarations  //
    ///////////////////////////

    //reset
    genvar g;
    logic aclrn, sclrn;                                 //these are the typical active low reset signals that are consumed
    logic sclrn_early_two, sclrn_early, sclrn_late;     //helpers for sclrn
    logic [RESET_DELAY_MAX:0] resetn_delayed;           //delayed versions of aclrn or sclrn, consumed by the occupancy trackers
    logic fifo_in_reset;                                //intended primarily for consumption by testbench to know when fifo is in reset, also used for stall_out when NEVER_OVERFLOWS=1

    //absorb excess earliness that the fifo cannot take advantage of
    logic [EXCESS_EARLY_STALL_PIPE:0] stall_in_pipe;
    logic [EXCESS_EARLY_VALID_PIPE:0] valid_in_pipe;
    logic stall_in_correct_timing, valid_in_correct_timing;

    //write control
    logic forced_write;
    logic [ADDR-1:0] wr_addr;

    //track the number of readable addressess
    logic wr_addr_ahead_of_rd_addr;

    //read control
    logic try_feed_prefetch, feed_prefetch;
    logic rd_addr_incr, prefetch_clock_en;
    logic [ADDR-1:0] rd_addr;



    /////////////
    //  Reset  //
    /////////////

    // S10 reset specification:
    // S (clocks to enter reset safe state) : 3 (sclrn_early_two -> sclrn_early -> resetn_delayed[*] -> stall_out),  beware synchronizer takes no time for reset assertion, but it does take time for reset release
    // P (minimum duration of reset pulse)  : 4 if RESET_EXTERNALLY_HELD = 1, otherwise 1 (we will internally pulse stretch the reset to 4 clocks)
    // D (clocks to exit reset safe state)  : 19 (3 for synchronizer) + (6 for sclrn_early_two to actual register) + (10 for reset release delay for registers that absorb excess earliness)

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
        assign sclrn_early = 1'b1;
        assign sclrn = 1'b1;
    end
    else begin : sync_reset
        logic [2:0] sclrn_chain;    //pulse extend from 1 clock to 4 clocks
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

    //this signal is consumed by the testbench to know whether the fifo is still in reset, can't use stall_out when fifo starts as full
    //this signal may be exported by the fifo in certain configurations, e.g. NEVER_OVERFLOWS=1
    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) fifo_in_reset <= 1'b1;
        else begin
            fifo_in_reset <= 1'b0;
            if (~resetn_delayed[RESET_DELAY]) fifo_in_reset <= 1'b1;
        end
    end



    ////////////////////////////////////////////////
    //  Absorb excess earliness on input signals  //
    ////////////////////////////////////////////////

    generate
    always_comb begin
        stall_in_pipe[0] = stall_in;
        valid_in_pipe[0] = valid_in;
    end
    for (g=1; g<=EXCESS_EARLY_STALL_PIPE; g++) begin : gen_stall_in_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) stall_in_pipe[g] <= 1'b1;
            else begin
                stall_in_pipe[g] <= stall_in_pipe[g-1];
                if (~sclrn && RESET_EVERYTHING) stall_in_pipe[g] <= 1'b1;
            end
        end
    end
    for (g=1; g<=EXCESS_EARLY_VALID_PIPE; g++) begin : gen_valid_in_delayed
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) valid_in_pipe[g] <= 1'b0;
            else begin
                valid_in_pipe[g] <= valid_in_pipe[g-1];
                if (~sclrn && RESET_EVERYTHING) valid_in_pipe[g] <= 1'b0;
            end
        end
    end
    endgenerate
    assign stall_in_correct_timing = stall_in_pipe[EXCESS_EARLY_STALL_PIPE];
    assign valid_in_correct_timing = valid_in_pipe[EXCESS_EARLY_VALID_PIPE];



    ////////////////////
    //  Memory block  //
    ////////////////////

    // Use MLAB in new data mode for mixed port read during write, unregister the read address within altdpram since this fifo will use ALM registers to drive the read address port.
    // Putting the read address register inside altdpram means we have no access to the output of the register, which is annoying since the fifo uses it as state, not just a pipeline stage.

    assign ecc_err_status = 2'h0;   // ECC IS NOT IMPLEMENTED YET, see case:555783
    generate
    if (WIDTH > 0) begin : GEN_MLAB
        altdpram #(     //modelsim library: altera_mf
            .indata_aclr                        ("OFF"),
            .indata_reg                         ("INCLOCK"),
            .intended_device_family             ("Stratix 10"),     //quartus will correct this automatically to whatever your project actually uses
            .lpm_type                           ("altdpram"),
            .ram_block_type                     ("MLAB"),
            .outdata_aclr                       ("OFF"),
            .outdata_sclr                       ("OFF"),
            .outdata_reg                        ("OUTCLOCK"),       //output data is registered, clock enable for this is controlled by outclocken
            .rdaddress_aclr                     ("OFF"),
            .rdaddress_reg                      ("UNREGISTERED"),   //we own the read address, bypass the equivalent of the internal address_b from m20k
            .rdcontrol_aclr                     ("OFF"),
            .rdcontrol_reg                      ("UNREGISTERED"),
            .read_during_write_mode_mixed_ports ("NEW_DATA"),       //new data mode allows lower latency through the fifo than mid speed fifo
            .width                              (WIDTH),
            .widthad                            (ADDR),
            .width_byteena                      (1),
            .wraddress_aclr                     ("OFF"),
            .wraddress_reg                      ("INCLOCK"),
            .wrcontrol_aclr                     ("OFF"),
            .wrcontrol_reg                      ("INCLOCK")
        )
        altdpram_component
        (
            //clock, no reset
            .inclock                            (clock),
            .outclock                           (clock),

            //write port
            .data                               (data_in),
            .wren                               (forced_write),
            .wraddress                          (wr_addr),

            //read port
            .rdaddress                          (rd_addr),
            .outclocken                         (prefetch_clock_en),
            .q                                  (data_out),

            //unused
            .aclr                               (1'b0),
            .sclr                               (1'b0),
            .byteena                            (1'b1),
            .inclocken                          (1'b1),
            .rdaddressstall                     (1'b0),
            .rden                               (1'b1),
            .wraddressstall                     (1'b0)
        );
    end
    endgenerate



    /////////////////////
    //  Write control  //
    /////////////////////

    assign forced_write = valid_in_correct_timing & ~stall_out;

    acl_lfsr #(
        .WIDTH                  (ADDR),
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_RESET      (0),
        .INITIAL_OCCUPANCY      (INITIAL_OCCUPANCY)
    )
    wr_addr_inst
    (
        .clock                  (clock),
        .resetn                 (resetn_delayed[0]),
        .enable                 (forced_write),
        .state                  (wr_addr)
    );



    ///////////////////////////////////////////////
    //  Track the number of readable addressess  //
    ///////////////////////////////////////////////

    // Once an address has been written to, it can also be read on the same clock cycle (mixed port read during write is new data). If we are advancing the write address now,
    // then we can also advance the read address now (assuming the read prefetch can catch the data). The occupancy tracker below tracks whether the read address is allowed
    // to advance because the write address has also advanced.

    acl_tessellated_incr_decr_threshold #(
        .CAPACITY                   (DEPTH),
        .THRESHOLD                  (1),
        .INITIAL_OCCUPANCY          (INITIAL_OCCUPANCY),
        .THRESHOLD_REACHED_AT_RESET (0),
        .WRITE_AND_READ_DURING_FULL (0),
        .WRITE_AND_READ_DURING_EMPTY(EARLY_STALL),
        .ASYNC_RESET                (ASYNC_RESET),
        .SYNCHRONIZE_RESET          (0),
        .RESET_EVERYTHING           (RESET_EVERYTHING),
        .RESET_EXTERNALLY_HELD      (1)
    )
    wr_addr_ahead_of_rd_addr_inst
    (
        .clock                      (clock),
        .resetn                     (resetn_delayed[RESET_DELAY_ADDR_COMPARE]),
        .incr_no_overflow           (forced_write),
        .incr_raw                   (forced_write),
        .decr_no_underflow          (feed_prefetch),
        .decr_raw                   (try_feed_prefetch),
        .threshold_reached          (wr_addr_ahead_of_rd_addr)
    );



    ////////////////////
    //  Read control  //
    ////////////////////

    generate
    if (EARLY_STALL == 0) begin : read_control_normal_timing
        assign try_feed_prefetch = ~valid_out | ~stall_in_correct_timing;       //will there be space in the read prefetch for new data to be captured?
        assign feed_prefetch = wr_addr_ahead_of_rd_addr & try_feed_prefetch;    //if so, is the read address allowed to advance because the write address has advanced?

        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                valid_out <= 1'b0;
            end
            else begin
                if (wr_addr_ahead_of_rd_addr & ~valid_out) valid_out <= 1'b1;                   //fifo becomes non-empty when prefetch is loaded
                if (~wr_addr_ahead_of_rd_addr & ~stall_in_correct_timing) valid_out <= 1'b0;    //fifo becomes empty when prefetch is read (and not reloaded)
                if (~sclrn) begin
                    valid_out <= 1'b0;
                end
            end
        end

        assign rd_addr_incr = feed_prefetch;
        assign prefetch_clock_en = (HOLD_DATA_OUT_WHEN_EMPTY) ? feed_prefetch : try_feed_prefetch;
        assign forced_read_out = valid_out & ~stall_in_correct_timing;
    end
    else begin : read_control_early_timing
        //if stall_in is 1 clock cycle early, manually retime the read side logic one clock cycle early
        //registering read control signals helps to mitigate against high fanout fmax loss

        //to retime wr_addr_ahead_of_rd_addr, we would need both the increment and decrement signals driving the occupancy tracker to be known one clock early
        //unfortunately we only have the decrement signal early
        //however keep in mind that wr_addr_ahead_of_rd_addr checks that the number of write address advances is not equal to the number of read address advances
        //we know that they would become unequal if the write address advances, so we use (wr_addr_ahead_of_rd_addr | forced_write) as the early retimed version

        logic valid_out_early;
        assign try_feed_prefetch = ~valid_out_early | ~stall_in_correct_timing;
        assign feed_prefetch = (wr_addr_ahead_of_rd_addr | forced_write) & try_feed_prefetch;

        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                valid_out_early <= 1'b0;
                valid_out <= 1'b0;
            end
            else begin
                if ((wr_addr_ahead_of_rd_addr | forced_write) & ~valid_out_early) valid_out_early <= 1'b1;
                if (~(wr_addr_ahead_of_rd_addr | forced_write) & ~stall_in_correct_timing) valid_out_early <= 1'b0;
                valid_out <= valid_out_early;
                if (~sclrn) begin
                    valid_out_early <= 1'b0;
                    valid_out <= 1'b0;
                end
            end
        end

        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                rd_addr_incr <= 1'b0;
                prefetch_clock_en <= 1'b0;
                forced_read_out <= 1'b0;
            end
            else begin
                rd_addr_incr <= feed_prefetch;
                prefetch_clock_en <= (HOLD_DATA_OUT_WHEN_EMPTY) ? feed_prefetch : try_feed_prefetch;
                forced_read_out <= valid_out_early & ~stall_in_correct_timing;
                if (~sclrn && RESET_EVERYTHING) begin
                    rd_addr_incr <= 1'b0;
                    prefetch_clock_en <= 1'b0;
                    forced_read_out <= 1'b0;
                end
            end
        end
    end
    endgenerate

    acl_lfsr #(
        .WIDTH                  (ADDR),
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_RESET      (0),
        .INITIAL_OCCUPANCY      (0)
    )
    rd_addr_inst
    (
        .clock                  (clock),
        .resetn                 (resetn_delayed[0]),
        .enable                 (rd_addr_incr),
        .state                  (rd_addr)
    );



    /////////////////
    //  Fifo full  //
    /////////////////

    generate
    if (NEVER_OVERFLOWS) begin : gen_reset_stall_out    //no overflow protection, but upstream still needs a way to know when fifo has exited from reset
        assign stall_out = fifo_in_reset;
    end
    else begin : gen_real_stall_out
        logic stall_out_raw;

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
            .resetn                     (resetn_delayed[RESET_DELAY_WRITE_USEDW]),
            .incr_no_overflow           (forced_write),
            .incr_raw                   (forced_write),
            .decr_no_underflow          (forced_read_out),
            .decr_raw                   (forced_read_out),
            .threshold_reached          (stall_out_raw)
        );

        assign stall_out = stall_out_raw & ((WRITE_AND_READ_DURING_FULL) ? ~forced_read_out : 1);
    end
    endgenerate



    ///////////////////
    //  Almost full  //
    ///////////////////

    generate
    if ((ALMOST_FULL_CUTOFF == 0) && (NEVER_OVERFLOWS == 0) && (WRITE_AND_READ_DURING_FULL == 0)) begin : full_almost_full
        assign almost_full = stall_out;
    end
    else begin : real_almost_full
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
            .resetn                     (resetn_delayed[RESET_DELAY_WRITE_USEDW]),
            .incr_no_overflow           (forced_write),
            .incr_raw                   (forced_write),
            .decr_no_underflow          (forced_read_out),
            .decr_raw                   (forced_read_out),
            .threshold_reached          (almost_full)
        );
    end
    endgenerate



    ////////////////////
    //  Almost empty  //
    ////////////////////

    generate
    if (ALMOST_EMPTY_CUTOFF == 0) begin : empty_almost_empty
        assign almost_empty = ~valid_out;
    end
    else begin : real_almost_empty
        logic not_almost_empty, forced_write_prev;

        //to track read used words, delay the fifo write signal by latency-1 clocks before driving the increment of the occupancy tracker
        //the occupancy tracker has a latency of 1 from its increment to threshold_reached, thus providing the correct write to almost empty latency
        //this fifo has a latency of 2, so delay the write by 1 clock cycle
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) forced_write_prev <= 1'b0;
            else begin
                forced_write_prev <= forced_write;
                if (~sclrn && RESET_EVERYTHING) forced_write_prev <= 1'b0;
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
            .resetn                     (resetn_delayed[RESET_DELAY_READ_USEDW]),
            .incr_no_overflow           (forced_write_prev),
            .incr_raw                   (forced_write_prev),
            .decr_no_underflow          (forced_read_out),
            .decr_raw                   (forced_read_out),
            .threshold_reached          (not_almost_empty)
        );
        assign almost_empty = ~not_almost_empty;
    end
    endgenerate

endmodule

`default_nettype wire
