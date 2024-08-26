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


// This module is the stall valid implementation of iowr. All interfaces must
// use same clock cycle handshaking.
//
// The basic operation of iowr is to transfer data from the kernel upstream
// interface to the channel facing interface when all the following are met:
// - kernel upstream data
// - channel has space available
// - kernel downstream has space available
//
// The actual implementation of iowr is not so straightforward. We can
// encounter deadlock situations if we directly connect iowr and iord blocks
// together in loop. For example:
//
// +-----------+      +-----------+
// |   iowr    |----->|   fifo    |
// +-----------+      |  (place   |
//      |             |  holder   |
//      |             | for some  |
//      V             |  other    |
// +-----------+      |  task or  |
// |   iord    |<-----|  kernel)  |
// +-----------+      +-----------+
//
// Assume everything is empty at power on. A valid arrives at the input of iowr.
// It is not accepted by iord since the fifo is empty (iord needs both upstreams
// to have a valid in order to make forward progress). Because the iowr cannot
// send a valid to iord, it also cannot send a valid to the fifo. Therefore the
// fifo will remain empty forever. This chicken-and-egg problem causes deadlocks.
//
// To solve this problem, iowr has something called "consumed registers". There are
// two consumers of iowr (the kernel downstream and the channel), and they are
// allowed to go at most one valid ahead of the other. This solves the chicken-and-egg
// problem. In the above example, one valid can be released to the fifo even though
// that valid has not been consumed by iord. This allows one item into the fifo,
// and therefore iord would eventually see a valid from the fifo and iowr.
//
// Inside iowr, the signals "consumed_sidepath" and "consumed_downstream"
// indicate that a valid has been allowed through on one path but not the other.
// If this happens, iowr will stall the kernel upstream, so the valid that was
// presented will stay there (and also the input data will be held). If the kernel
// downstream went ahead of the channel, then consumed_downstream will turn on, which
// also prevents the same valid from being presented to kernel downstream again.
// Likewise by symmetry consumed_sidepath means the channnel went ahead.
//
// To support multiple write sites to the same channel, we need to ensure that
// consumed_downstream never turns on. We must avoid the scenario where iowr
// did not write to the channel but releases a valid to kernel downstream, because
// the valid released to kernel downstream could reach another iowr and this
// second iowr could write to the same channel first.
//
// Beware: when ACK_AS_VALID = 1, it is not known if the one-sided consumed register
// avoids deadlock in general. Deadlock will not occur if the concurrency limit is 1.

`default_nettype none

//refer to hld_iowr.sv for a description of the parameters and ports

module hld_iowr_stall_valid #(
    //core spec
    parameter int DATA_WIDTH,
    parameter bit NON_BLOCKING,

    //reset
    parameter bit ASYNC_RESET,
    parameter bit SYNCHRONIZE_RESET,

    //downstream
    parameter bit DISCONNECT_DOWNSTREAM,
    parameter bit ENABLED,
    parameter bit ACK_AS_VALID
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
    output logic                    o_ack,

    //sidepath
    output logic                    o_fifovalid,
    input  wire                     i_fifostall,
    output logic   [DATA_WIDTH-1:0] o_fifodata
);

    logic aclrn, sclrn, resetn_synchronized;
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


    //no changes to data path
    assign o_fifodata = i_data;

    generate
    if (DISCONNECT_DOWNSTREAM) begin : GEN_BYPASS
        //connect upstream to sidepath directly
        assign o_stall     = i_fifostall;
        assign o_fifovalid = i_valid;
        assign o_ack       = 1'b1;
        assign o_valid     = 1'b1;
    end
    else if (NON_BLOCKING) begin : GEN_NON_BLOCKING
        assign o_valid     = i_valid;
        assign o_stall     = i_stall;
        assign o_fifovalid = i_valid & ~i_predicate & ~i_stall;
        assign o_ack       = o_fifovalid & ~i_fifostall;
    end
    else begin : GEN_BLOCKING
        logic consumed_downstream, consumed_sidepath;
        logic stall_contribution_downstream, stall_contribution_sidepath;
        logic clock_enable;

        assign clock_enable = (ENABLED) ? ~i_stall : 1'b1;  //ENABLED=1 means this iowr is inside a stall enable cluster

        always_ff @(posedge clock or negedge aclrn) begin
            if (~aclrn) begin
                consumed_downstream <= 1'b0;
                consumed_sidepath <= 1'b0;
            end
            else begin
                if (clock_enable) begin
                    consumed_downstream <= o_stall & stall_contribution_downstream;
                    consumed_sidepath   <= o_stall & stall_contribution_sidepath;
                end
                if (~sclrn) begin
                    consumed_downstream <= 1'b0;
                    consumed_sidepath <= 1'b0;
                end
            end
        end

        if (ACK_AS_VALID) begin : GEN_ACK_AS_VALID  //one-sided consumed registers for multiple writes to the same channel
            assign stall_contribution_downstream = 1'b0;
            assign o_stall = i_stall | (i_fifostall & ~i_predicate & ~consumed_sidepath);
            assign o_valid = (o_fifovalid & ~i_fifostall) | (i_valid & i_predicate) | consumed_sidepath;
        end
        else begin : NO_ACK_AS_VALID    //normal behavior of iowr (each of the two consumers have a consumed register)
            assign stall_contribution_downstream = consumed_downstream | (i_valid & ~i_stall);
            assign o_stall = ~stall_contribution_downstream | ~stall_contribution_sidepath;
            assign o_valid = i_valid & ~consumed_downstream;
        end
        assign stall_contribution_sidepath = consumed_sidepath | (i_valid & (i_predicate | ~i_fifostall));
        assign o_fifovalid = i_valid & ~i_predicate & ~consumed_sidepath;
    end
    endgenerate

endmodule

`default_nettype wire
