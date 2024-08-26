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


// acl_staging_reg.sv
//
// Module to implement a staging register.  Used to pipeline stall signals.
//

`default_nettype none

module acl_staging_reg
(
    clk, reset, i_data, i_valid, o_stall, o_data, o_valid, i_stall
);

/*************
* Parameters *
*************/
parameter WIDTH=32;
parameter bit ASYNC_RESET = 1;          // how do the registers CONSUME reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
parameter bit SYNCHRONIZE_RESET = 0;    // before consumption, do we SYNCHRONIZE the reset: 1 means use a synchronizer (assume reset arrived asynchronously), 0 means passthrough (assume reset was already synchronized)

/********
* Ports *
********/
// Standard global signals
input wire clk;
input wire reset;

// Upstream interface
input wire [WIDTH-1:0] i_data;
input wire i_valid;
output logic o_stall;

// Downstream interface
output logic [WIDTH-1:0] o_data;
output logic o_valid;
input wire i_stall;

/***************
* Architecture *
***************/
logic [WIDTH-1:0] r_data;
logic r_valid;

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

// Upstream
assign o_stall = r_valid;

// Downstream
assign o_data = (r_valid) ? r_data : i_data;
assign o_valid = (r_valid) ? r_valid : i_valid;

// Storage reg
always@(posedge clk or negedge aclrn)
begin
    if(~aclrn)
    begin
        r_valid <= 1'b0;
        r_data <= 'x;   // don't need to reset
    end
    else
    begin
        if (~r_valid) r_data <= i_data;
        r_valid <= i_stall && (r_valid || i_valid);
        if(~sclrn)
        begin
            r_valid <= 1'b0;
            r_data <= 'x;   // don't need to reset
        end
    end
end

endmodule

`default_nettype wire
