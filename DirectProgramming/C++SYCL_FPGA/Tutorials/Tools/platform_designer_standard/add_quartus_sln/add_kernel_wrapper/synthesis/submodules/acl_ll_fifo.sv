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


// Low latency FIFO
// One cycle latency from all inputs to all outputs
// Storage implemented in registers, not memory.

// altera message_off 10036

`default_nettype none

module acl_ll_fifo(clk, reset, data_in, write, data_out, read, empty, full, almost_full);

/* Parameters */
parameter WIDTH = 32;
parameter DEPTH = 32;
parameter ALMOST_FULL_VALUE = 0;
parameter bit ASYNC_RESET = 1;          // how do the registers CONSUME reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
parameter bit SYNCHRONIZE_RESET = 0;    // before consumption, do we SYNCHRONIZE the reset: 1 means use a synchronizer (assume reset arrived asynchronously), 0 means passthrough (assume reset was already synchronized)

/* Ports */
input wire clk;
input wire reset;
input wire [WIDTH-1:0] data_in;
input wire write;
output logic [WIDTH-1:0] data_out;
input wire read;
output logic empty;
output logic full;
output logic almost_full;

/* Architecture */
// One-hot write-pointer bit (indicates next position to write at),
// last bit indicates the FIFO is full
logic [DEPTH:0] wptr;
// Replicated copy of the stall / valid logic
logic [DEPTH:0] wptr_copy /* synthesis dont_merge */;
// FIFO data registers
logic [DEPTH-1:0][WIDTH-1:0] data;

// Write pointer updates:
logic wptr_hold; // Hold the value
logic wptr_dir;  // Direction to shift

// Data register updates:
logic [DEPTH-1:0] data_hold;     // Hold the value
logic [DEPTH-1:0] data_new;      // Write the new data value in

// Reset
logic aclrn, sclrn;
acl_reset_handler
#(
    .ASYNC_RESET            (ASYNC_RESET),
    .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
    .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
    .PULSE_EXTENSION        (0),
    .PIPE_DEPTH             (1),
    .NUM_COPIES             (1)
)
acl_reset_handler_inst
(
    .clk                    (clk),
    .i_resetn               (~reset),
    .o_aclrn                (aclrn),
    .o_resetn_synchronized  (),
    .o_sclrn                (sclrn)
);

// Write location is constant unless the occupancy changes
assign wptr_hold = !(read ^ write);
assign wptr_dir = read;

// Hold the value unless we are reading, or writing to this
// location
genvar i;
generate
for(i = 0; i < DEPTH; i++)
begin : data_mux
    assign data_hold[i] = !(read | (write & wptr[i]));
    assign data_new[i] = !read | wptr[i+1];
end
endgenerate

// The data registers
generate
for(i = 0; i < DEPTH-1; i++)
begin : data_reg
    always@(posedge clk or negedge aclrn)
    begin
        if(~aclrn) begin
            data[i] <= {WIDTH{1'b0}};
        end
        else begin
            data[i] <= data_hold[i] ? data[i] : data_new[i] ? data_in : data[i+1];
            if (~sclrn) data[i] <= {WIDTH{1'b0}};
        end
    end
end
endgenerate
always@(posedge clk or negedge aclrn)
begin
    if(~aclrn) begin
        data[DEPTH-1] <= {WIDTH{1'b0}};
    end
    else begin
        data[DEPTH-1] <= data_hold[DEPTH-1] ? data[DEPTH-1] : data_in;
        if (~sclrn) data[DEPTH-1] <= {WIDTH{1'b0}}; //TODO this is probably not necessary, but need to make sure we don't have any legacy circuits that depend on data being 0 at power up for example
    end
end

// The write pointer
always@(posedge clk or negedge aclrn)
begin
    if(~aclrn) begin
        wptr <= {{DEPTH{1'b0}}, 1'b1};
        wptr_copy <= {{DEPTH{1'b0}}, 1'b1};
    end
    else begin
        wptr <= wptr_hold ? wptr : wptr_dir ? {1'b0, wptr[DEPTH:1]} : {wptr[DEPTH-1:0], 1'b0};
        wptr_copy <= wptr_hold ? wptr_copy : wptr_dir ? {1'b0, wptr_copy[DEPTH:1]} : {wptr_copy[DEPTH-1:0], 1'b0};
        if(~sclrn) begin
            wptr <= {{DEPTH{1'b0}}, 1'b1};
            wptr_copy <= {{DEPTH{1'b0}}, 1'b1};
        end
    end
end

// Outputs
assign empty = wptr_copy[0];
assign full = wptr_copy[DEPTH];
assign almost_full = |wptr_copy[DEPTH:ALMOST_FULL_VALUE];
assign data_out = data[0];

endmodule

`default_nettype wire
