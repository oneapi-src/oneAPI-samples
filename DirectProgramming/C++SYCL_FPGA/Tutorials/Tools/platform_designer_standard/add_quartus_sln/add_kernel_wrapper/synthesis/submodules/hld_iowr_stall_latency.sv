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


// This module is the stall latency implementation of iowr. One may choose
// whether to use stall latency or stall valid on each interface. Note that even
// if a design uses stall latency, if the iowr is non-blocking, often (but not
// always) the compiler will put it inside a stall free cluster, in which case
// the stall valid implementation would be used.
//
// The basic operation of iowr is to transfer data from the kernel upstream
// interface to the channel facing interface when all the following are met:
// - kernel upstream data
// - channel has space available
// - kernel downstream has space available
//
// The high level operation of iowr is the same between the stall valid and
// stall latency implementations, however the stall latency verion must provide
// capacity at each interface. Technically it is legal to choose capacity=0 for
// each interface (stall latency implementation acts as stall valid), however
// the FIFOs for capacity are not removed so this would be a huge area overhead
// versus using the stall valid implementation.
//
// Stall latency requires capacity for capturing transactions already in flight,
// however with the proper architecture this capacity can be shared between
// interfaces and across modules. The strategies used can be found in this
// powerpoint:
// p4/depot/docs/hld/ip/fifo_merging_across_channels.pptx

`default_nettype none

//refer to hld_iowr.sv for a description of the parameters and ports

module hld_iowr_stall_latency
// hld_memory_depth_quantization_pkg should be imported here; however, there is
// an issue with the simulator that causes a crash on HLS multi-component
// designs when SV package files are imported. Case 14017832227 tracks restoring
// that import once the underlying issues have been resolved.
#(
    //core spec
    parameter int DATA_WIDTH,
    parameter bit NON_BLOCKING,

    //reset
    parameter bit ASYNC_RESET,
    parameter bit SYNCHRONIZE_RESET,

    //upstream
    parameter bit USE_STALL_LATENCY_UPSTREAM,
    parameter int ALMOST_FULL_CUTOFF_UPSTREAM,
    parameter bit NO_PREDICATION,

    //downstream
    parameter bit USE_STALL_LATENCY_DOWNSTREAM,
    parameter int ALMOST_EMPTY_CUTOFF_DOWNSTREAM,
    parameter int STALL_IN_EARLINESS_DOWNSTREAM,

    //sidepath
    parameter bit USE_STALL_LATENCY_SIDEPATH,
    parameter int ALMOST_FULL_CUTOFF_SIDEPATH,
    parameter bit DISCONNECT_DOWNSTREAM,
    parameter int CAPACITY_FROM_CHANNEL,
    parameter int INTER_KERNEL_PIPELINING,

    //other features
    parameter bit ALLOW_HIGH_SPEED_FIFO_USAGE,
    parameter     enable_ecc
) (
    input  wire                     clock,
    input  wire                     resetn,

    //upstream
    input  wire                     i_valid,
    input  wire                     i_predicate,
    output logic                    o_stall,
    input  wire    [DATA_WIDTH-1:0] i_data,

    //downstream
    output logic                    o_valid,
    input  wire                     i_stall,
    output logic                    o_empty,
    output logic                    o_almost_empty,
    output logic                    o_ack,

    //sidepath
    output logic                    o_fifovalid,
    input  wire                     i_fifostall,
    input  wire                     i_fifochannel_stall,
    output logic   [DATA_WIDTH-1:0] o_fifodata,

    //other
    output logic              [1:0] ecc_err_status,
    output logic                    o_profile_o_stall,
    output logic                    o_profile_fifo_stall,
    output logic             [31:0] o_channel_fifo_occupancy
);

    // The following two functions should be imported from
    // hld_memory_depth_quantization_pkg.sv; however, there is an issue with the
    // simulator that causes a crash on HLS multi-component designs when SV
    // package files are imported. Case 14017832227 tracks restoring that import
    // and removing the function copies from this file once the underlying
    // issues have been resolved.

    // If the M20K is narrow, can use a deeper depth.
    function automatic int quantizeRamDepthUsingWidth;
    input int depth, width;
    begin
        quantizeRamDepthUsingWidth =
            (depth <= 32)                  ?                    32 :    //fits into min depth MLAB
            (depth <= 2048 && width <= 10) ?                  2048 :    //fits into single M20K
            (depth <= 1024 && width <= 20) ?                  1024 :    //fits into single M20K
            (depth <= 512)                 ?                   512 :    //fits into min depth M20K
                                             ((depth+511)/512)*512 ;    //round up to nearest multiple of 512
    end
    endfunction

    // Same idea as above, but use a width-aware depth quantization.
    function automatic int quantizeFifoDepthUsingWidth;
    input int depth, width;
    begin
        quantizeFifoDepthUsingWidth = 2 ** $clog2(quantizeRamDepthUsingWidth(depth, width));
    end
    endfunction

    ////////////////////////////////////////////////////
    //  Determine the depth and latency of each fifo  //
    ////////////////////////////////////////////////////

    //properties of the fifo or occupancy tracker
    //in cases where a fifo is zero width, reduce both area and latency by using the occupancy tracker directly
    localparam     HLD_FIFO_STYLE           = (ALLOW_HIGH_SPEED_FIFO_USAGE) ? "hs" : "ms";
    localparam int HLD_FIFO_LATENCY         = (ALLOW_HIGH_SPEED_FIFO_USAGE) ? 5 : 3;
    localparam int OCC_TRACKER_LATENCY      = 1;

    //side fifo receives data from kernel upstream and provides it to the sidepath
    localparam int SIDE_FIFO_LATENCY        = HLD_FIFO_LATENCY + INTER_KERNEL_PIPELINING;           //effective latency through the fifo after factoring in the pipelining in front of hld_fifo
    localparam int SIDE_FIFO_DEPTH_DESIRED  = CAPACITY_FROM_CHANNEL                                 //capacity consolidated from channel
                                            + ALMOST_FULL_CUTOFF_UPSTREAM                           //catch up to this many transactions already in flight
                                            + 1                                                     //almost full is used regardless of predication, add 1 to capacity to ensure predicated threads can make progress
                                            + 1 + ALMOST_FULL_CUTOFF_UPSTREAM + SIDE_FIFO_LATENCY;  //ensure no bubbles, latency of i_stall -> o_stall -> i_valid -> o_valid
    localparam int SIDE_FIFO_DEPTH          = quantizeFifoDepthUsingWidth(SIDE_FIFO_DEPTH_DESIRED, DATA_WIDTH);

    //down fifo indicates whether the sidepath was written to in nonblocking mode, fifo width is zero in blocking mode
    localparam int DOWN_FIFO_LATENCY        = (NON_BLOCKING) ? HLD_FIFO_LATENCY : OCC_TRACKER_LATENCY;
    localparam int DOWN_FIFO_DEPTH_DESIRED  = ALMOST_EMPTY_CUTOFF_DOWNSTREAM                        //if there is a sync downstream of this block, need this much occupancy to keep it in fast read mode
                                            + ALMOST_FULL_CUTOFF_UPSTREAM                           //catch up to this many transactions already in flight
                                            + 1 + ALMOST_FULL_CUTOFF_UPSTREAM + DOWN_FIFO_LATENCY;  //ensure no bubbles, latency of i_stall -> o_stall -> i_valid -> o_valid
    localparam int DOWN_FIFO_DEPTH          = quantizeFifoDepthUsingWidth(DOWN_FIFO_DEPTH_DESIRED, 1);

    //can bypass the sidepath fifo if iowr is blocking, and upstream and sidepath speak the same protocol wtih the same amount of early backpressure
    localparam bit BYPASS_SIDE_FIFO         = !NON_BLOCKING && (USE_STALL_LATENCY_SIDEPATH == USE_STALL_LATENCY_UPSTREAM) && (ALMOST_FULL_CUTOFF_SIDEPATH == ALMOST_FULL_CUTOFF_UPSTREAM);


    ///////////////
    //  Signals  //
    ///////////////

    //reset
    logic aclrn, sclrn, resetn_synchronized;

    //error correction code status
    logic [1:0] side_ecc, down_ecc;

    //transact
    logic up_incr, side_transact, side_almost_full, side_somewhat_full;
    logic side_read, delayed_side_transact, side_fifo_empty;
    logic [DATA_WIDTH-1:0] delayed_i_data;

    //downstream
    logic delayed_i_stall, down_decr, down_almost_full, down_has_one, down_has_many, down_somewhat_full;

    //profiler
    logic profile_side_stall, profile_down_stall;



    /////////////
    //  Reset  //
    /////////////

    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
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



    //error correction code status
    assign ecc_err_status = side_ecc | down_ecc;



    ////////////////
    //  Transact  //
    ////////////////

    // The basic strategy is to collect almost full from the two consumers, OR them together, and then pass that
    // to kernel upstream to let it decide whether a transaction should occur. To deal with nonblocking mode and
    // predication, mask the transaction from upstream for any consumer who should not see that transaction.

    assign up_incr = i_valid & ((USE_STALL_LATENCY_UPSTREAM) ? 1'b1 : ~o_stall);
    assign o_stall = down_almost_full | ((NON_BLOCKING) ? 1'b0 : side_almost_full);
    assign side_transact = up_incr & ((NO_PREDICATION) ? 1'b1 : ~i_predicate) & ((NON_BLOCKING) ? ~side_almost_full : 1'b1);

    generate
    if (BYPASS_SIDE_FIFO) begin : NO_SIDE_FIFO
        assign o_fifovalid        = side_transact;
        assign o_fifodata         = i_data;
        assign side_almost_full   = i_fifostall;
        assign side_ecc           = '0;
        assign profile_side_stall = i_fifochannel_stall;    //use profiling information supplied by iord, capacity from side_fifo has been moved into iord
    end
    else begin : GEN_SIDE_FIFO
        assign side_read = (USE_STALL_LATENCY_SIDEPATH) ? o_fifovalid : o_fifovalid & ~i_fifostall;

        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (SIDE_FIFO_DEPTH),
            .THRESHOLD                  (SIDE_FIFO_DEPTH - ((NON_BLOCKING) ? 0 : ALMOST_FULL_CUTOFF_UPSTREAM)),
            .THRESHOLD_REACHED_AT_RESET (1),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0)
        )
        side_almost_full_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_synchronized),
            .incr_no_overflow           (side_transact),
            .incr_raw                   (side_transact),
            .decr_no_underflow          (side_read),
            .decr_raw                   (side_read),
            .threshold_reached          (side_almost_full)
        );

        //inter kernel pipelining -- add some pipelining stages before the fifo
        acl_shift_register_no_reset #(
            .WIDTH      (DATA_WIDTH+1),
            .STAGES     (INTER_KERNEL_PIPELINING)
        )
        delay_write_interface_of_side_fifo
        (
            .clock      (clock),
            .D          ({i_data, side_transact}),
            .Q          ({delayed_i_data, delayed_side_transact})
        );

        hld_fifo
        #(
            .WIDTH                          (DATA_WIDTH),
            .DEPTH                          (SIDE_FIFO_DEPTH),
            .ASYNC_RESET                    (ASYNC_RESET),
            .SYNCHRONIZE_RESET              (0),
            .STYLE                          (HLD_FIFO_STYLE),
            .USE_STALL_LATENCY_UPSTREAM     (1),
            .NEVER_OVERFLOWS                (1),
            .USE_STALL_LATENCY_DOWNSTREAM   (USE_STALL_LATENCY_SIDEPATH),
            .STALL_IN_EARLINESS             (ALMOST_FULL_CUTOFF_SIDEPATH),
            .enable_ecc                     (enable_ecc)
        )
        side_fifo
        (
            .clock                          (clock),
            .resetn                         (resetn_synchronized),
            .i_valid                        (delayed_side_transact),
            .i_data                         (delayed_i_data),
            .o_valid                        (o_fifovalid),
            .o_empty                        (side_fifo_empty),
            .o_data                         (o_fifodata),
            .i_stall                        (i_fifostall),
            .ecc_err_status                 (side_ecc)
        );

        //profiler
        if (CAPACITY_FROM_CHANNEL) begin : GEN_PROFILER_FIFO_MERGE
            //iowr implements the channel only when iord has no capacity (iord is either missing or is nonblocking and put inside stall free cluster)
            //side_fifo implements the capacity for both iowr upstream catching transactions already in flight and the channel capacity
            //use the same somewhat full strategy that iord would have implemented if the channel were implemented inside iord
            acl_tessellated_incr_decr_threshold #(
                .CAPACITY                   (SIDE_FIFO_DEPTH),
                .THRESHOLD                  (SIDE_FIFO_DEPTH - (2*ALMOST_FULL_CUTOFF_UPSTREAM+1)),
                .THRESHOLD_REACHED_AT_RESET (1),
                .ASYNC_RESET                (ASYNC_RESET),
                .SYNCHRONIZE_RESET          (0)
            )
            side_somewhat_full_inst
            (
                .clock                      (clock),
                .resetn                     (resetn_synchronized),
                .incr_no_overflow           (side_transact),
                .incr_raw                   (side_transact),
                .decr_no_underflow          (side_read),
                .decr_raw                   (side_read),
                .threshold_reached          (side_somewhat_full)
            );
            assign profile_side_stall = side_somewhat_full & ~side_read;
        end
        else begin : GEN_PROFILER_NO_MERGE
            //the channel fifo (implemented in acl_channel_fifo or acl_stream_fifo) is a physically separate from the upstream fifo (implemented here as side_fifo)
            //use the strategy from the old stall latency implementation: upstream not empty and channel full
            assign profile_side_stall = ~side_fifo_empty & i_fifostall;     //upstream has work accumulated in side fifo but channel is full
        end
    end
    endgenerate



    //////////////////
    //  Downstream  //
    //////////////////

    // Nonblocking mode has to produce o_ack which indicates whether the write to the channel succeeded. Blocking mode has no
    // such requirement, but instead of using a zero width fifo, can reduce area and latency by directly using the occupancy tracker.

    //stall in earliness is not all that useful for occ trackers, just absorb it
    acl_shift_register_no_reset #(.WIDTH(1), .STAGES(STALL_IN_EARLINESS_DOWNSTREAM)) delayed_i_stall_inst (.clock(clock), .D(i_stall), .Q(delayed_i_stall));

    generate
    if (DISCONNECT_DOWNSTREAM) begin : GEN_DOWN_DISCONNECT
        assign o_valid = 1'b1;
        assign o_ack = 1'b1;
        assign o_empty = 1'b0;
        assign o_almost_empty = 1'b0;
        assign down_almost_full = 1'b0;
        assign down_ecc = '0;
    end
    else if (NON_BLOCKING) begin : GEN_DOWN_FIFO
        hld_fifo
        #(
            .WIDTH                          (1),
            .DEPTH                          (DOWN_FIFO_DEPTH),
            .ASYNC_RESET                    (ASYNC_RESET),
            .SYNCHRONIZE_RESET              (0),
            .STYLE                          (HLD_FIFO_STYLE),
            .ALMOST_FULL_CUTOFF             (ALMOST_FULL_CUTOFF_UPSTREAM),
            .USE_STALL_LATENCY_UPSTREAM     (1),
            .NEVER_OVERFLOWS                (1),
            .ALMOST_EMPTY_CUTOFF            (ALMOST_EMPTY_CUTOFF_DOWNSTREAM),
            .USE_STALL_LATENCY_DOWNSTREAM   (USE_STALL_LATENCY_DOWNSTREAM),
            .STALL_IN_EARLINESS             (STALL_IN_EARLINESS_DOWNSTREAM),
            .enable_ecc                     (enable_ecc)
        )
        down_fifo
        (
            .clock                          (clock),
            .resetn                         (resetn_synchronized),
            .i_valid                        (up_incr),
            .i_data                         (side_transact),
            .o_stall                        (down_almost_full), //in stall latency mode, o_stall is actually an almost full
            .o_valid                        (o_valid),
            .o_empty                        (o_empty),
            .o_almost_empty                 (o_almost_empty),
            .o_data                         (o_ack),
            .i_stall                        (i_stall),
            .ecc_err_status                 (down_ecc)
        );
    end
    else begin : GEN_DOWN_OCC
        assign down_decr = ~o_empty & ~delayed_i_stall;
        assign down_ecc = '0;

        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DOWN_FIFO_DEPTH),
            .THRESHOLD                  (DOWN_FIFO_DEPTH - ALMOST_FULL_CUTOFF_UPSTREAM),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0)
        )
        down_almost_full_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_synchronized),
            .incr_no_overflow           (up_incr),
            .incr_raw                   (up_incr),
            .decr_no_underflow          (down_decr),
            .decr_raw                   (down_decr),
            .threshold_reached          (down_almost_full)
        );

        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DOWN_FIFO_DEPTH),
            .THRESHOLD                  (1),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0)
        )
        down_has_one_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_synchronized),
            .incr_no_overflow           (up_incr),
            .incr_raw                   (up_incr),
            .decr_no_underflow          (down_decr),
            .decr_raw                   (down_decr),
            .threshold_reached          (down_has_one)
        );

        acl_tessellated_incr_decr_threshold #(
            .CAPACITY                   (DOWN_FIFO_DEPTH),
            .THRESHOLD                  (ALMOST_EMPTY_CUTOFF_DOWNSTREAM + 1),
            .ASYNC_RESET                (ASYNC_RESET),
            .SYNCHRONIZE_RESET          (0)
        )
        down_has_many_inst
        (
            .clock                      (clock),
            .resetn                     (resetn_synchronized),
            .incr_no_overflow           (up_incr),
            .incr_raw                   (up_incr),
            .decr_no_underflow          (down_decr),
            .decr_raw                   (down_decr),
            .threshold_reached          (down_has_many)
        );

        //output signals
        assign o_valid = (USE_STALL_LATENCY_DOWNSTREAM) ? down_decr : down_has_one;
        assign o_empty = ~down_has_one;
        assign o_almost_empty = ~down_has_many;
    end
    endgenerate
    
    // This is only used by the profiler. If upstream produces data faster than downstream can consume it, the steady state
    // behavior an osscillation of many clock cycles of accepting valids and many clock cycles of the being almost full.
    // The profiler is interested in whether upstream has data to send which can be interpreted in a few different ways:
    // - downstream not empty: upstream has work queued in this fifo, however this interpretation does not take into account the
    //     depth of the down fifo which adds elasticity to the data path
    // - downstream almost full: this interpretation takes into account the down fifo depth, however it suffers from the swings in
    //     almost full which doesn't accurately reflect upstream always wants to send data when it is faster than downstream
    // - downstream somewhat full: a similar approach to almost full (which takes into account the down fifo depth) but lower the
    //     threshold enough not be impacted by the swings in almost full
    // Somewhat full is the best interpretation to use.
    acl_tessellated_incr_decr_threshold #(
        .CAPACITY                   (DOWN_FIFO_DEPTH),
        .THRESHOLD                  (DOWN_FIFO_DEPTH - (2*ALMOST_FULL_CUTOFF_UPSTREAM+1)),
        .ASYNC_RESET                (ASYNC_RESET),
        .SYNCHRONIZE_RESET          (0)
    )
    down_somewhat_full_inst
    (
        .clock                      (clock),
        .resetn                     (resetn_synchronized),
        .incr_no_overflow           (up_incr),
        .incr_raw                   (up_incr),
        .decr_no_underflow          (down_decr),
        .decr_raw                   (down_decr),
        .threshold_reached          (down_somewhat_full)
    );
    assign profile_down_stall = down_somewhat_full;



    ////////////////
    //  Profiler  //
    ////////////////

    // Note that all of the logic below will have no fanout (will be synthesized away) unless ACL_PROFILE = 1.
    //
    // If fifo merging decides to implement the channel fifo inside iord, expose how many words are inside the fifo to the profiler. Want the binary count of
    // occupancy, so can't use the occupancy tracker. To mitigate against fmax loss, register the increment and decrement, so the occupancy is slightly stale.

    generate
    if (CAPACITY_FROM_CHANNEL) begin : GEN_CHANNEL_FIFO_OCCUPANCY
        logic incr, decr;
        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                incr <= 1'b0;
                decr <= 1'b0;
                o_channel_fifo_occupancy <= '0;
            end
            else begin
                incr <= delayed_side_transact;  //write into fifo
                decr <= side_read;              //read from fifo
                o_channel_fifo_occupancy <= o_channel_fifo_occupancy + incr - decr; //how many words are inside the fifo
                if (~sclrn) begin
                    incr <= 1'b0;
                    decr <= 1'b0;
                    o_channel_fifo_occupancy <= '0;
                end
            end
        end
    end
    else begin : NO_CHANNEL_FIFO_OCCUPANCY
        assign o_channel_fifo_occupancy = '0;
    end
    endgenerate

    // The profiler is currently tapping signals that fundamentally do not exist in the stall latency protocol. For example, one thing the profiler is interested
    // in is when the channel stalls the upstream interface. For iowr, this means upstream wants to send a valid but cannot since the channel is full. This
    // requires upstream to be able to say "work is available and may be consumed", however stall latency only allows upstream to force a transaction. This can
    // be worked around by adding a FIFO after the upstream interface but before core iowr logic which decides whether a transaction will proceed. Without FIFO
    // merging, we can look at the read side of side_fifo to check if upstream has queued work which is not making progress. If FIFO merging moves the capacity of
    // iowr into iord, fundamentally this information is still available but now the read side of the FIFO lives inside iord. To resolve this, iord implements
    // profiling logic for iowr, and a new signal (channel_stall) is routed from iord through channel to iowr (corresponding changes have been made in
    // Griffin/XNodeExternalIP and in system integrator).

    always_ff @(posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            o_profile_o_stall <= 1'b0;
            o_profile_fifo_stall <= 1'b0;
        end
        else begin
            o_profile_o_stall <= profile_side_stall | profile_down_stall;   //upstream wants to send a valid but iowr is stalling (channel full or downstream full)
            o_profile_fifo_stall <= profile_side_stall;                     //upstream wants to send a nonpredicated valid but channel is full
            if (~sclrn) begin
                o_profile_o_stall <= 1'b0;
                o_profile_fifo_stall <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
