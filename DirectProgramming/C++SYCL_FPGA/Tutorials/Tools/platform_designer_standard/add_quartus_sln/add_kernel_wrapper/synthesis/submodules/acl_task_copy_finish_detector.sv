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



// This module detects when a task kernel copy has finished. It asserts
// kernel_copy_finished when valid_out for the copy has been higher and
// the copy has no pending writes.

`default_nettype none

module acl_task_copy_finish_detector #(
    parameter int ASYNC_RESET = 1,          // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter int SYNCHRONIZE_RESET = 0,    // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
    parameter int PIPELINE_VALID_OUT = 1    // do not use the un-registered input valid signal to calculate kernel done to synchronize timing with kernel_copy_has_pending_write input; the latter input gets registered at the output of each LSU
)
(
    input wire clock,
    input wire start,
    input wire resetn,
    input wire kernel_copy_valid_out,
    input wire kernel_copy_has_pending_write,

    output reg kernel_copy_finished
);
    
    logic aclrn, sclrn;
    acl_reset_handler
    #(
        .ASYNC_RESET            (ASYNC_RESET),
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),
        .PIPE_DEPTH             (1),
        .NUM_COPIES             (1)
    )
    acl_reset_handler_inst
    (
        .clk                    (clock),
        .i_resetn               (resetn),
        .o_aclrn                (aclrn),
        .o_resetn_synchronized  (),
        .o_sclrn                (sclrn)
    );
    
    // Stores whether or not valid_out has been high in the past
    reg valid_out_has_been_high;

    always @ (posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            valid_out_has_been_high <= 1'b0;
        end
        else begin
            if (start) begin
                valid_out_has_been_high <= 1'b0;
            end
            else if (kernel_copy_valid_out) begin
                valid_out_has_been_high <= 1'b1;
            end
            if (~sclrn) begin
                valid_out_has_been_high <= 1'b0;
            end
        end
    end

    always @ (posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            kernel_copy_finished <= 1'b0;
        end
        else begin
            if (start) begin
                kernel_copy_finished <= 1'b0;
            end
            else if (((PIPELINE_VALID_OUT==0 && kernel_copy_valid_out) | valid_out_has_been_high) & ~kernel_copy_has_pending_write) begin
                kernel_copy_finished <= 1'b1;
            end
            if (~sclrn) begin
                kernel_copy_finished <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
