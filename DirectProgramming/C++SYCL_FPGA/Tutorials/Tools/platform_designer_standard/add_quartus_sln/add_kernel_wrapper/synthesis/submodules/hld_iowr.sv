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


// The HLD compiler uses hld_iowr to perform a write to a channel/stream/pipe.
// The basic operation of hld_iowr is to transfer data from the kernel upstream
// interface to the channel facing interface when all the following are met:
// - kernel upstream data
// - channel has space available
// - kernel downstream has space available
//
// This module is a wrapper around the stall/valid and stall latency
// implementations. This wrapper deals with the following:
// - profiling: monitor for backpressure and data starvation
// - cutpaths: tie off the channel interface, this allows one to measure fmax
//     for one kernel in isolation even though that kernel requires channels
// - data packing: bundle data path signals together (data, startofpacket,
//     endofpacket, packetempty)

`include "acl_parameter_assert.svh"
`default_nettype none

module hld_iowr #(
    //core specification
    parameter int DATA_WIDTH = 32,                      //width of the data path
    parameter int NON_BLOCKING = 0,                     //non-blocking means a valid can still go from kernel upstream to downstream without writing to channel if channel is full

    //reset
    parameter bit ASYNC_RESET = 1,                      //should registers consume reset asynchronously (ASYNC_RESET=1) or synchronously (ASYNC_RESET=0)
    parameter bit SYNCHRONIZE_RESET = 0,                //before consumption by registers, should the reset be synchronized to the clock

    //upstream configuration
    parameter bit USE_STALL_LATENCY_UPSTREAM = 0,       //0 means stall valid (valid means transaction may happen but doesn't have to), 1 means use stall latency (valid means forced transaction)
    parameter int ALMOST_FULL_CUTOFF_UPSTREAM = 0,      //specify how early the backpressure to kernel upstream needs to be, note the use of hld_fifo cutoff semantics, this is NOT almost full threshold
    parameter int NO_PREDICATION = 0,                   //set to 1 if it is known that i_predicate is a constant 0, the IP is simpler and smaller without predication support

    //downstream configuration
    parameter bit USE_STALL_LATENCY_DOWNSTREAM = 0,     //0 means stall valid, 1 means stall latency
    parameter int ALMOST_EMPTY_CUTOFF_DOWNSTREAM = 0,   //specify how early the starvation to kernel downstream needs to be
    parameter int STALL_IN_EARLINESS_DOWNSTREAM = 0,    //stall in earliness for the fifo that feeds that to kernel downstream
    parameter bit DISCONNECT_DOWNSTREAM = 0,            //HLS launch/collect capacity balancing: remove the dependency if it is covered through the task being launched and collect
    parameter bit ENABLED = 0,                          //if hld_iowr is used inside a stall enable cluster, also need to freeze the consumed registers
    parameter bit ACK_AS_VALID = 0,                     //if multiple writes to the same channel in user code, channel write happens before releasing a valid to kernel downstream, modify the consumed registers

    //sidepath configuration
    parameter bit USE_STALL_LATENCY_SIDEPATH = 0,       //0 means stall valid, 1 means stall latency
    parameter int ALMOST_FULL_CUTOFF_SIDEPATH = 0,      //how early the backpressure is from either the channel itself or iord on the other side of the channel
    parameter int CAPACITY_FROM_CHANNEL = 0,            //consolidation of capacity across iowr, channel, and iord -- how much capacity should this module absorb from others
    parameter int INTER_KERNEL_PIPELINING = 0,          //how much pipelining to add before the single fifo that holds all of the capacity for iowr, channel and iord

    //other features
    parameter int EMPTY_WIDTH = 0,                      //width of the empty signal for avalon packet data
    parameter bit CUTPATHS = 0,                         //tie off channel interface, intended for fmax measurement of one kernel in isolation but that kernel uses channels
    parameter bit ACL_PROFILE = 0,                      //enable the profiler
    parameter ACL_PROFILE_SHARED=0,                     // Set to 1 to enable shared counters
    parameter bit ALLOW_HIGH_SPEED_FIFO_USAGE = 1,      //choice of hld_fifo style, 0 = mid speed fifo, 1 = high speed fifo
    parameter     enable_ecc = "FALSE",                 //enable error correction codes for RAM, legal values are "FALSE" or "TRUE"
    
    //derived values
    parameter int EMPTY_PORT = (EMPTY_WIDTH > 0) ? EMPTY_WIDTH : 1

) (
    input  wire                     clock,
    input  wire                     resetn,

    //upstream control
    input  wire                     i_predicate,        //0 means write to the channel, 1 means don't write to the channel (which means downstream may see a valid even if channel is full)
    input  wire                     i_valid,            //if USE_STALL_LATENCY_UPSTREAM = 1 this means we must accept the transaction from upstream, else this means upstream has a transaction which we may accept
    output logic                    o_stall,            //backpressure to upstream

    //upstream data
    input  wire    [DATA_WIDTH-1:0] i_data,
    input  wire                     i_startofpacket,
    input  wire                     i_endofpacket,
    input  wire              [31:0] i_packetempty,

    //downstream control
    output logic                    o_valid,            //if USE_STALL_LATENCY_DOWNSTREAM = 1 this means downstream must accept the transaction, else this means there is a transaction for downstream which it may accept
    input  wire                     i_stall,            //backpressure from downstream, earliness is determined by STALL_IN_EARLINESS_DOWNSTREAM
    output logic                    o_ack,              //only used in non-blocking, indicates whether data was written to sidepath
    output logic                    o_empty,            //indicates to downstream that there are no transactions available
    output logic                    o_almost_empty,     //will be stuck at 1 if IP has no capacity

    //sidepath control
    output logic                    o_fifovalid,        //if USE_STALL_LATENCY_SIDEPATH = 1 this means sidepath must accept the transaction, else this means there is a transaction for sidepath which it may accept
    input  wire                     i_fifoready,        //backpressure from sidepath
    input  wire                     i_fifochannel_stall,//if iowr capacity merges into iord, iord tells iowr when the channel interface would have been stalling upstream, intended for profiler

    //sidepath data
    output logic   [DATA_WIDTH-1:0] o_fifodata,
    output logic                    o_fifostartofpacket,
    output logic                    o_fifoendofpacket,
    output logic   [EMPTY_PORT-1:0] o_fifoempty,

    //others
    output logic              [1:0] ecc_err_status,     //error correction code status
    input  wire              [31:0] i_fifosize,         //profiler port descriptions are below in the profiler section
    input  wire               [1:0] profile_shared_control, // control what the profiler shared counters are counting
    output logic                    profile_shared,         // counting profile_i_valid if control is 0, profile_fifo_stall if control is 1, profile_idle if control is 2, and profile_total_req if control is 3
    output logic                    profile_i_valid,
    output logic                    profile_i_stall,
    output logic                    profile_o_stall,
    output logic                    profile_idle,
    output logic                    profile_total_req,
    output logic                    profile_fifo_stall,
    output logic                    profile_total_fifo_size,
    output logic             [31:0] profile_total_fifo_size_incr,
    output logic             [31:0] o_fifosize
);
    ///////////////////////////////////////
    // Parameter checking
    //
    // Generate an error if any illegal parameter settings or combinations are used
    ///////////////////////////////////////
    initial /* synthesis enable_verilog_initial_construct */
    begin
        if (EMPTY_PORT != ((EMPTY_WIDTH > 0) ? EMPTY_WIDTH : 1))
            $fatal(1, "Illegal parameteriazation, EMPTY_PORT must be not be specified when instantiating this module, it must be left at its default value");
    end

    localparam bit USE_STALL_LATENCY_IP = USE_STALL_LATENCY_UPSTREAM || USE_STALL_LATENCY_DOWNSTREAM || USE_STALL_LATENCY_SIDEPATH;
    logic stall_latency_profile_o_stall, stall_latency_profile_fifo_stall;
    logic [31:0] channel_fifo_occupancy;



    ///////////////////////////////////
    //  Simulation-only debug hooks  //
    ///////////////////////////////////

    // to assist in debug of surrounding logic outside of hld_iowr -- signals of interest are named sim_only_debug_***
    // technically this is synthesizable logic, but it would degrade fmax and it is not hooked up to anything
    // synthesis translate_off
    int sim_only_debug_upstream_count, sim_only_debug_upstream_nonpredicated_count, sim_only_debug_downstream_count, sim_only_debug_sidepath_count;
    int sim_only_debug_upstream_to_downstream_occupancy, sim_only_debug_upstream_nonpredicated_to_sidepath_occupancy;
    logic sim_only_debug_upstream_incr, sim_only_debug_upstream_nonpredicated_incr, sim_only_debug_downstream_incr, sim_only_debug_sidepath_incr;

    genvar g;
    logic [STALL_IN_EARLINESS_DOWNSTREAM:0] pipe_i_stall;
    logic correct_timing_i_stall;
    generate
    always_comb begin
        pipe_i_stall[0] = i_stall;
    end
    for (g=1; g<=STALL_IN_EARLINESS_DOWNSTREAM; g++) begin : GEN_RANDOM_BLOCK_NAME_R83
        always_ff @(posedge clock) begin
            pipe_i_stall[g] <= pipe_i_stall[g-1];
        end
    end
    endgenerate
    assign correct_timing_i_stall = pipe_i_stall[STALL_IN_EARLINESS_DOWNSTREAM];

    //determine whether a transaction has occurred at various boundary locations in the IP
    assign sim_only_debug_upstream_incr = (USE_STALL_LATENCY_UPSTREAM) ? i_valid : i_valid & ~o_stall;
    assign sim_only_debug_upstream_nonpredicated_incr = sim_only_debug_upstream_incr & ~i_predicate;
    assign sim_only_debug_downstream_incr = ~o_empty & ~correct_timing_i_stall;
    assign sim_only_debug_sidepath_incr = (USE_STALL_LATENCY_SIDEPATH) ? o_fifovalid : o_fifovalid & i_fifoready;

    //keep track of the transaction count at various boundary locations in the IP
    always_ff @(posedge clock or negedge resetn) begin
        if (~resetn) begin
            sim_only_debug_upstream_count <= '0;
            sim_only_debug_upstream_nonpredicated_count <= '0;
            sim_only_debug_downstream_count <= '0;
            sim_only_debug_sidepath_count <= '0;
        end
        else begin
            sim_only_debug_upstream_count <= sim_only_debug_upstream_count + sim_only_debug_upstream_incr;
            sim_only_debug_upstream_nonpredicated_count <= sim_only_debug_upstream_nonpredicated_count + sim_only_debug_upstream_nonpredicated_incr;
            sim_only_debug_downstream_count <= sim_only_debug_downstream_count + sim_only_debug_downstream_incr;
            sim_only_debug_sidepath_count <= sim_only_debug_sidepath_count + sim_only_debug_sidepath_incr;
        end
    end

    //how many transactions have entered upstream but not yet exited downstream
    assign sim_only_debug_upstream_to_downstream_occupancy = sim_only_debug_upstream_count - sim_only_debug_downstream_count;

    //how many nonpredicated transactions have entered upstream but not yet exited from sidepath
    assign sim_only_debug_upstream_nonpredicated_to_sidepath_occupancy = sim_only_debug_upstream_nonpredicated_count - sim_only_debug_sidepath_count;
    // synthesis translate_on



    ////////////////
    //  Profiler  //
    ////////////////

    generate
    if (ACL_PROFILE) begin : GEN_PROFILE
        //there is a fundamental limitation of the stall latency protocol which makes things challenging for the profiler
        //upstream of iowr can only communicate that it is forcing a valid, it cannot say that there is work available which may be accepted
        //by looking only at the input/output ports, it is impossible to infer that upstream wants to send work but cannot since the channel is full
        //if there were a fifo between the upstream interface and the core iowr logic that decides whether a transaction proceeds, one could look at the read side of this fifo, which indicates if work is available
        //without fifo merging this fifo lives inside iowr, however if fifo merging moves the capacity of iowr into iord, this information is still fundamentally available but now it lives inside iord
        //to make the profiler work, iord sends this info through the channel and eventually to iowr as a new signal called channel_stall
        
        logic delayed_i_stall;  //remove earliness on i_stall
        acl_shift_register_no_reset #(.WIDTH(1), .STAGES(STALL_IN_EARLINESS_DOWNSTREAM)) delayed_i_stall_inst (.clock(clock), .D(i_stall), .Q(delayed_i_stall));

        assign profile_i_valid = (USE_STALL_LATENCY_UPSTREAM) ? i_valid : (i_valid & ~o_stall);                             //upstream has sent a valid to us
        assign profile_i_stall = ~o_empty & delayed_i_stall;                                                                //we want to send a valid but downstream is stalling
        assign profile_o_stall = (USE_STALL_LATENCY_UPSTREAM) ? stall_latency_profile_o_stall : (i_valid & o_stall);        //upstream wants to send a valid but we are stalling
        assign profile_total_req = profile_i_valid & ~i_predicate;                                                          //upstream has sent a nonpredicated valid to us
        assign profile_idle = ( ~profile_i_valid & ~profile_fifo_stall );
        
        if (NON_BLOCKING) begin
            assign profile_fifo_stall  = '0; // cannot stall in a non-blocking write
        end else begin
            assign profile_fifo_stall = (USE_STALL_LATENCY_UPSTREAM) ? stall_latency_profile_fifo_stall : ~i_fifoready & i_valid & ~i_predicate;    // upstream wants to send a nonpredicated valid but channel is full
        end
        
        // Count channel depth when the iowr is either writing, or being stalled by the channel (ie. it is interacting with the channel)
        assign profile_total_fifo_size = (USE_STALL_LATENCY_SIDEPATH) ? (o_fifovalid | profile_fifo_stall) : ((o_fifovalid & i_fifoready) | profile_fifo_stall);
        assign profile_total_fifo_size_incr = i_fifosize;                                                                   // increment by channel depth amounts
        assign o_fifosize = (USE_STALL_LATENCY_IP) ? channel_fifo_occupancy : '0;                                           //how many words are inside the channel fifo if implemented inside iowr

        if (ACL_PROFILE_SHARED == 1)
        begin
            always@(posedge clock) begin
                case (profile_shared_control)
                    2'b00: profile_shared <= profile_i_valid;
                    2'b01: profile_shared <= profile_fifo_stall;
                    2'b10: profile_shared <= profile_idle;
                    2'b11: profile_shared <= profile_total_req;
                    default: profile_shared <= 1'b0;
                endcase
            end
        end
    end
    else begin : NO_PROFILE
        assign profile_i_valid = '0;
        assign profile_i_stall = '0;
        assign profile_o_stall = '0;
        assign profile_idle = '0;
        assign profile_total_req = '0;
        assign profile_fifo_stall = '0;
        assign profile_total_fifo_size = '0;
        assign profile_total_fifo_size_incr = '0;
        assign o_fifosize = '0;
        assign profile_shared = '0;
    end
    endgenerate



    ///////////////////////////////////
    //  Intercept channel interface  //
    ///////////////////////////////////

    logic                   ch_o_fifovalid;
    logic  [DATA_WIDTH-1:0] ch_o_fifodata;
    logic                   ch_i_fifostall;
    logic                   ch_i_fifochannel_stall;
    logic                   ch_o_fifostartofpacket;
    logic                   ch_o_fifoendofpacket;
    logic  [EMPTY_PORT-1:0] ch_o_fifoempty;

    generate
    if (CUTPATHS) begin : GEN_CUTPATHS
        logic                   virt_o_fifovalid         /* synthesis dont_merge keep preserve noprune */;
        logic  [DATA_WIDTH-1:0] virt_o_fifodata          /* synthesis dont_merge keep preserve noprune */;
        logic                   virt_i_fifostall         /* synthesis dont_merge keep preserve noprune */;
        logic                   virt_i_fifochannel_stall /* synthesis dont_merge keep preserve noprune */;
        logic                   virt_o_fifostartofpacket /* synthesis dont_merge keep preserve noprune */;
        logic                   virt_o_fifoendofpacket   /* synthesis dont_merge keep preserve noprune */;
        logic  [EMPTY_PORT-1:0] virt_o_fifoempty         /* synthesis dont_merge keep preserve noprune */;

        //fake inputs
        always_ff @(posedge clock) begin
            virt_i_fifostall         <= ~virt_i_fifostall;
            virt_i_fifochannel_stall <= ~virt_i_fifochannel_stall;
        end
        assign ch_i_fifostall         = virt_i_fifostall;
        assign ch_i_fifochannel_stall = virt_i_fifochannel_stall;

        //fake outputs
        always_ff @(posedge clock) begin
            virt_o_fifovalid         <= ch_o_fifovalid;
            virt_o_fifodata          <= ch_o_fifodata;
            virt_o_fifostartofpacket <= ch_o_fifostartofpacket;
            virt_o_fifoendofpacket   <= ch_o_fifoendofpacket;
            virt_o_fifoempty         <= ch_o_fifoempty;
        end
    end
    else begin : NO_CUTPATHS
        assign ch_i_fifostall         = ~i_fifoready;
        assign ch_i_fifochannel_stall = i_fifochannel_stall;
        assign o_fifovalid            = ch_o_fifovalid;
        assign o_fifodata             = ch_o_fifodata;
        assign o_fifostartofpacket    = ch_o_fifostartofpacket;
        assign o_fifoendofpacket      = ch_o_fifoendofpacket;
        assign o_fifoempty            = ch_o_fifoempty;
    end
    endgenerate



    //////////////////////////////////
    //  Choose iowr implementation  //
    //////////////////////////////////

    generate
    if (USE_STALL_LATENCY_IP) begin : GEN_STALL_LATENCY

        //parameter legality checks -- ensure stall valid features are not used here
        `ACL_PARAMETER_ASSERT(ENABLED == 0)
        `ACL_PARAMETER_ASSERT(ACK_AS_VALID == 0)

        hld_iowr_stall_latency
        #(
            //core spec
            .DATA_WIDTH                     (EMPTY_WIDTH + 2 + DATA_WIDTH),
            .NON_BLOCKING                   (NON_BLOCKING),

            //reset
            .ASYNC_RESET                    (ASYNC_RESET),
            .SYNCHRONIZE_RESET              (SYNCHRONIZE_RESET),

            //upstream
            .USE_STALL_LATENCY_UPSTREAM     (USE_STALL_LATENCY_UPSTREAM),
            .ALMOST_FULL_CUTOFF_UPSTREAM    (ALMOST_FULL_CUTOFF_UPSTREAM),
            .NO_PREDICATION                 (NO_PREDICATION),

            //downstream
            .USE_STALL_LATENCY_DOWNSTREAM   (USE_STALL_LATENCY_DOWNSTREAM),
            .ALMOST_EMPTY_CUTOFF_DOWNSTREAM (ALMOST_EMPTY_CUTOFF_DOWNSTREAM),
            .STALL_IN_EARLINESS_DOWNSTREAM  (STALL_IN_EARLINESS_DOWNSTREAM),

            //sidepath
            .USE_STALL_LATENCY_SIDEPATH     (USE_STALL_LATENCY_SIDEPATH),
            .ALMOST_FULL_CUTOFF_SIDEPATH    (ALMOST_FULL_CUTOFF_SIDEPATH),
            .DISCONNECT_DOWNSTREAM          (DISCONNECT_DOWNSTREAM),
            .CAPACITY_FROM_CHANNEL          (CAPACITY_FROM_CHANNEL),
            .INTER_KERNEL_PIPELINING        (INTER_KERNEL_PIPELINING),

            //other features
            .ALLOW_HIGH_SPEED_FIFO_USAGE    (ALLOW_HIGH_SPEED_FIFO_USAGE),
            .enable_ecc                     (enable_ecc)
        )
        hld_iowr_stall_latency_inst
        (
            .clock                          (clock),
            .resetn                         (resetn),

            //upstream
            .i_valid                        (i_valid),
            .i_predicate                    (i_predicate),
            .o_stall                        (o_stall),
            .i_data                         ({i_packetempty[EMPTY_PORT-1:0], i_endofpacket, i_startofpacket, i_data}),

            //downstream
            .o_valid                        (o_valid),
            .i_stall                        (i_stall),
            .o_empty                        (o_empty),
            .o_almost_empty                 (o_almost_empty),
            .o_ack                          (o_ack),

            //channel
            .o_fifovalid                    (ch_o_fifovalid),
            .i_fifostall                    (ch_i_fifostall),
            .i_fifochannel_stall            (ch_i_fifochannel_stall),
            .o_fifodata                     ({ch_o_fifoempty, ch_o_fifoendofpacket, ch_o_fifostartofpacket, ch_o_fifodata}),

            //other
            .ecc_err_status                 (ecc_err_status),
            .o_profile_o_stall              (stall_latency_profile_o_stall),
            .o_profile_fifo_stall           (stall_latency_profile_fifo_stall),
            .o_channel_fifo_occupancy       (channel_fifo_occupancy)
        );
    end
    else begin : GEN_STALL_VALID

        //parameter legality checks -- ensure stall latency features are not used here
        `ACL_PARAMETER_ASSERT(ALMOST_FULL_CUTOFF_UPSTREAM == 0)
        `ACL_PARAMETER_ASSERT(ALMOST_EMPTY_CUTOFF_DOWNSTREAM == 0)
        `ACL_PARAMETER_ASSERT(STALL_IN_EARLINESS_DOWNSTREAM == 0)
        `ACL_PARAMETER_ASSERT(ALMOST_FULL_CUTOFF_SIDEPATH == 0)
        `ACL_PARAMETER_ASSERT(CAPACITY_FROM_CHANNEL == 0)
        `ACL_PARAMETER_ASSERT(INTER_KERNEL_PIPELINING == 0)

        hld_iowr_stall_valid
        #(
            //core spec
            .DATA_WIDTH                     (EMPTY_WIDTH + 2 + DATA_WIDTH),
            .NON_BLOCKING                   (NON_BLOCKING),

            //reset
            .ASYNC_RESET                    (ASYNC_RESET),
            .SYNCHRONIZE_RESET              (SYNCHRONIZE_RESET),

            //downstream
            .DISCONNECT_DOWNSTREAM          (DISCONNECT_DOWNSTREAM),
            .ENABLED                        (ENABLED),
            .ACK_AS_VALID                   (ACK_AS_VALID)
        )
        hld_iowr_stall_valid_inst
        (
            .clock                          (clock),
            .resetn                         (resetn),

            //upstream
            .i_valid                        (i_valid),
            .i_predicate                    (i_predicate),
            .o_stall                        (o_stall),
            .i_data                         ({i_packetempty[EMPTY_PORT-1:0], i_endofpacket, i_startofpacket, i_data}),

            //downstream
            .o_valid                        (o_valid),
            .i_stall                        (i_stall),
            .o_ack                          (o_ack),

            //channel
            .o_fifovalid                    (ch_o_fifovalid),
            .i_fifostall                    (ch_i_fifostall),
            .o_fifodata                     ({ch_o_fifoempty, ch_o_fifoendofpacket, ch_o_fifostartofpacket, ch_o_fifodata})
        );
        assign o_empty = ~o_valid;
        assign o_almost_empty = 1'b1;
        assign ecc_err_status = '0;
    end
    endgenerate

endmodule

`default_nettype wire
