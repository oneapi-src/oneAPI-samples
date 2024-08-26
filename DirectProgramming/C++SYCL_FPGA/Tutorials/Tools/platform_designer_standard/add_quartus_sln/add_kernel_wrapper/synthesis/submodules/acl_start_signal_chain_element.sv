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


// The start signal chain element forwards the start signal received 
// from the previous chain element to the kernel copy it serves as well as
// the corresponding task copy finish detector and finish chain element.

`default_nettype none

module acl_start_signal_chain_element #(
    parameter int ASYNC_RESET = 1,          // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter int SYNCHRONIZE_RESET = 0     // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
)
(
    input wire clock,
    input wire resetn,

    input wire start_in,
    output reg start_kernel,
    output reg start_finish_detector,
    output reg start_finish_chain_element,
    output reg start_chain
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
    
    always @ (posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            start_chain <= 1'b0;
        end
        else begin
            start_chain <= start_in;
            if (~sclrn) begin
                start_chain <= 1'b0;
            end
        end
    end

    always @ (posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            start_kernel <= 1'b0;
            start_finish_detector <= 1'b0;
            start_finish_chain_element <= 1'b0;
        end
        else begin
            start_kernel <= start_chain;
            start_finish_detector <= start_chain;
            start_finish_chain_element <= start_chain;
            if (~sclrn) begin
                start_kernel <= 1'b0;
                start_finish_detector <= 1'b0;
                start_finish_chain_element <= 1'b0;
            end
        end
    end


endmodule

`default_nettype wire
