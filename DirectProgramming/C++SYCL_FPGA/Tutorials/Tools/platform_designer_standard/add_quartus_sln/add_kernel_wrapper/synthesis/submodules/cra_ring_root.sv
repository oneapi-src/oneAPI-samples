// (c) 1992-2024 Intel Corporation.                            
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


`default_nettype none

module cra_ring_root #(
    parameter integer ASYNC_RESET = 1,          // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter integer SYNCHRONIZE_RESET = 0,    // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    parameter integer ADDR_W = 32,
    parameter integer DATA_W = 32,
    parameter integer ID_W = 3,
    parameter integer ROM_EXT_W = 0,
    parameter integer ROM_ENABLE = 0,
    parameter integer AGENT_PORT_WAITREQUEST_ALLOWANCE = 0,// 0 means the Avalon waitrequest-allowance feature is disabled (ie. we use immediate backpressure). >0 means the feature is enabled and this module will accept this many requests after avs_waitrequest asserts.
    parameter integer ALLOW_HIGH_SPEED_FIFO_USAGE = 1,      // choice of hld_fifo style, 0 = mid speed fifo, 1 = high speed fifo
    parameter integer ZERO_LATENCY = 0
)
(
    // clock/reset
    input wire clk,
    input wire rst_n,

    // avalon-agent port
    input wire avs_enable, // dummy ports : for bind_port compatibility with AVS signals in the System Integrator
    input wire avs_read,
    input wire avs_write,
    input wire [ADDR_W+ID_W+ROM_EXT_W+ROM_ENABLE-1:0] avs_addr, // If the width of avs_addr ever changes, RING_ADDR_WIDTH must be updated.
    input wire [DATA_W/8-1:0] avs_byteena,
    input wire [DATA_W-1:0] avs_writedata,
    output logic [DATA_W-1:0] avs_readdata,
    output logic avs_readdatavalid,
    output logic avs_waitrequest,

    // ring-in
    input wire ri_read,                         // Not consumed
    input wire ri_write,                        // Not consumed
    input wire [ADDR_W+ID_W-1:0] ri_addr,       // Not consumed
    input wire [DATA_W-1:0] ri_data,
    input wire [DATA_W-1:0] ri_readdata,        // not consumed if ENABLE_ZERO_LATENCY=0
    input wire [DATA_W/8-1:0] ri_byteena,       // Not consumed
    input wire ri_datavalid,

    // ring-out
    output logic ro_read,
    output logic ro_write,
    output logic [ADDR_W+ID_W+ROM_EXT_W+ROM_ENABLE-1:0] ro_addr,
    output logic [DATA_W-1:0] ro_data,
    output logic [DATA_W-1:0] ro_readdata,       // not consumed if ENABLE_ZERO_LATENCY=0
    output logic [DATA_W/8-1:0] ro_byteena,
    output logic ro_datavalid
);

localparam RING_ADDR_WIDTH = ADDR_W+ID_W+ROM_EXT_W+ROM_ENABLE;
localparam ENABLE_WRA = AGENT_PORT_WAITREQUEST_ALLOWANCE > 0; // Enable waitrequest allowance
localparam ENABLE_ZERO_LATENCY = ZERO_LATENCY && !ENABLE_WRA; // can't be zero latency if we are using a fifo

logic aclrn, sclrn, resetn_synchronized;
acl_reset_handler
#(
    .ASYNC_RESET            (ASYNC_RESET),
    .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
    .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
    .PIPE_DEPTH             (3),
    .NUM_COPIES             (1)
)
acl_reset_handler_inst
(
    .clk                    (clk),
    .i_resetn               (rst_n),
    .o_aclrn                (aclrn),
    .o_resetn_synchronized  (resetn_synchronized),
    .o_sclrn                (sclrn)
);

logic pending;
logic avs_fifo_o_stall, avs_fifo_i_stall, avs_fifo_o_valid, avs_fifo_o_data_read, avs_fifo_o_data_write, avs_fifo_o_empty;
logic [RING_ADDR_WIDTH-1:0] avs_fifo_o_data_addr;
logic [DATA_W/8-1:0] avs_fifo_o_data_byteena;
logic [DATA_W-1:0] avs_fifo_o_data_writedata;

generate
    if(ENABLE_ZERO_LATENCY) begin : GEN_ZERO_LATENCY_READDATA
        assign avs_readdatavalid = ri_datavalid;
        assign avs_readdata = ri_readdata;
    end
    else begin
        // The avalon agent connection
        always@(posedge clk or negedge aclrn) begin
            if (~aclrn) begin
                avs_readdatavalid <= 1'b0;
                avs_readdata <= 'x;
            end
            else begin
                avs_readdatavalid <= ri_datavalid;
                avs_readdata <= ri_data;
                if (~sclrn) begin
                    avs_readdatavalid <= 1'b0;
                end
            end
        end
    end
endgenerate


generate
    if (ENABLE_WRA) begin
        /*
            If waitrequest-allowance on the avs (Avalon agent) interface is disabled then we generate backpressure on the Avalon interface whenever a read is pending on the CRA ring. The backpressure is immediate (ie. upstream must cease sending
                requests on the same cycle, which is the normal Avalon waitrequest behaviour). Backpressure is released when the read data is returned. Avalon requests have a max burst size of 1 on this interface.
            If waitrequest-allowance is enabled we instantiate a FIFO to buffer incoming requests from the kernel-interface in the BSP. Backpressure is generated using the FIFO's almost-full and
                incoming requests may continue for a few cycles. This is done to allow avs_waitrequest to be pipelined by the upstream interface, for performance.
                We still use the concept of pending-reads, meaning we only pull new requests from the FIFO when there no pending read requests on the ring.
                We also use the stall-in-earliness feature of hld-fifo which effectively increases the read latency to 4. We attempt to pull a new request every 5 cycles and only pause
                if we successfully pull a read request ('pending' means to pause as such).
        */

        // Store the full request in the avs_fifo: read + write + address + write_data + byte_en
        localparam AVS_FIFO_WIDTH = 1 + 1 + RING_ADDR_WIDTH + DATA_W + (DATA_W/8);

        hld_fifo
        #(
            .WIDTH                          (AVS_FIFO_WIDTH),
            .DEPTH                          (AGENT_PORT_WAITREQUEST_ALLOWANCE+8),   // Arbitrarily selecting AGENT_PORT_WAITREQUEST_ALLOWANCE+8. We just need a small FIFO.
            .ALMOST_FULL_CUTOFF             (AGENT_PORT_WAITREQUEST_ALLOWANCE),
            .ASYNC_RESET                    (ASYNC_RESET),
            .SYNCHRONIZE_RESET              (0),    // Reset is synchronized by this module's reset-handler.
            .STALL_IN_EARLINESS             (0),    // Set to 0 because when we backpressure the fifo we need it to stop outputting valids since we have no capacity downstream of the FIFO.
            .NEVER_OVERFLOWS                (0),
            .USE_STALL_LATENCY_UPSTREAM     (1),
            .USE_STALL_LATENCY_DOWNSTREAM   (0),    // Similar to above, we need zero-cycle handshaking (no capacity downstream)
            .STYLE                          (ALLOW_HIGH_SPEED_FIFO_USAGE ? "hs" : "ms")
        )
        avs_fifo
        (
            .clock                          (clk),
            .resetn                         (resetn_synchronized),

            // Upstream interface
            .i_valid                        (avs_read || avs_write),
            .i_data                         ({avs_read, avs_write, avs_addr, avs_writedata, avs_byteena}),
            .o_stall                        (avs_fifo_o_stall), // This is the almost-full if USE_STALL_LATENCY_UPSTREAM == 1
            .o_almost_full                  (), // Not used if USE_STALL_LATENCY_UPSTREAM == 1

            // Downstream interface
            .o_valid                        (avs_fifo_o_valid),
            .o_data                         ({avs_fifo_o_data_read, avs_fifo_o_data_write, avs_fifo_o_data_addr, avs_fifo_o_data_writedata, avs_fifo_o_data_byteena}),
            .i_stall                        (avs_fifo_i_stall),
            .o_almost_empty                 (),
            .o_empty                        (avs_fifo_o_empty)
        );

        assign avs_fifo_i_stall = avs_fifo_o_empty || pending; // Read whenever the FIFO has data, but don't read if we're pending.

    end else begin
        assign avs_fifo_o_valid = 1'b0;
        assign avs_fifo_o_data_read = 1'b0;
        assign avs_fifo_o_stall = 1'b0;
    end
endgenerate

// case:199865 (scheduled for clean-up in case:200564): ISR concurrent to the
// main thread may issue a request immediately following a read.  The CRA ring
// is not latency balanced so this can lead to a conflict on the shared data
// bus.  Quick fix: each read blocks until the response is sent back.

// Assert pending when a read-request is issued to the ring. De-assert only when the read data comes back.
always@(posedge clk or negedge aclrn) begin
    if (~aclrn) begin
        pending <= 1'b0;
    end
    else begin
        pending <= pending ?    (!ri_datavalid) :
                                ENABLE_WRA? (avs_fifo_o_valid && !avs_fifo_i_stall && avs_fifo_o_data_read): avs_read;
        if (~sclrn) begin
            pending <= 1'b0;
        end
    end
end

generate
    if(ENABLE_ZERO_LATENCY) begin : GEN_ZERO_LATENCY_WAITREQUEST
        assign avs_waitrequest = ri_datavalid ? 1'b0 : pending;
    end
    else begin
        assign avs_waitrequest = ENABLE_WRA ? avs_fifo_o_stall : pending;
    end
endgenerate

generate
    if(ENABLE_ZERO_LATENCY) begin : GEN_ZERO_LATENCY_RO
        assign ro_read = avs_read && !avs_waitrequest;
        assign ro_write = avs_write && !avs_waitrequest;
        assign ro_addr = avs_addr;
        assign ro_data = avs_writedata;
        assign ro_readdata = {DATA_W{1'b0}};              // ring root doesn't generate read data
        assign ro_byteena = avs_byteena;
        assign ro_datavalid = 1'b0;
    end
    else begin
        // The ring output
        always@(posedge clk or negedge aclrn) begin
            if (~aclrn) begin
                ro_read <= 1'b0;
                ro_write <= 1'b0;
                ro_datavalid <= 1'b0;
                ro_addr <= 'x;
                ro_data <= 'x;
                ro_byteena <= 'x;
            end
            else begin

                if (ENABLE_WRA) begin
                    ro_read <= avs_fifo_o_data_read && avs_fifo_o_valid && !avs_fifo_i_stall;
                    ro_write <= avs_fifo_o_data_write && avs_fifo_o_valid && !avs_fifo_i_stall;
                    ro_addr <= avs_fifo_o_data_addr;
                    ro_data <= avs_fifo_o_data_writedata;
                    ro_byteena <= avs_fifo_o_data_byteena;
                end else begin
                    ro_read <= avs_read && !avs_waitrequest;
                    ro_write <= avs_write && !avs_waitrequest;
                    ro_addr <= avs_addr;
                    ro_data <= avs_writedata;
                    ro_byteena <= avs_byteena;
                end

                ro_datavalid <= 1'b0;

                if (~sclrn) begin
                    ro_read <= 1'b0;
                    ro_write <= 1'b0;
                    ro_datavalid <= 1'b0;
                end
            end
        end
    end
endgenerate
endmodule

`default_nettype wire
