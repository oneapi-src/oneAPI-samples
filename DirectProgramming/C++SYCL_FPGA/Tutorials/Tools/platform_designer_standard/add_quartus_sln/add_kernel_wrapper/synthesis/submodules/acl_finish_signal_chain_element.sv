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


// The final signal chain element generates generates a one-cycle
// pulse on finish_out when the copy it serves has finished and 
// the element before it in the chain generates a pulse
// on the finish_in input.

`default_nettype none

module acl_finish_signal_chain_element #(
    parameter int ASYNC_RESET = 1,          // how do we use reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter int SYNCHRONIZE_RESET = 0     // based on how reset gets to us, what do we need to do: 1 means synchronize reset before consumption (if reset arrives asynchronously), 0 means passthrough (managed externally)
)
(
    input wire clock,
    input wire resetn,
    input wire start,

    input wire kernel_copy_finished,

    input wire finish_in,
    output reg finish_out
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
    
    reg prev_has_finished;
    reg finish_out_asserted;

    always @ (posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            prev_has_finished <= 1'b0;
        end
        else begin
            if ( start ) begin
                prev_has_finished <= 1'b0;
            end
            else if ( finish_in ) begin
                prev_has_finished <= 1'b1;
            end
            if (~sclrn) begin
                prev_has_finished <= 1'b0;
            end
        end
    end

    always @ (posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            finish_out <= 1'b0;
        end
        else begin
            finish_out <= kernel_copy_finished & prev_has_finished & ~finish_out_asserted;
            if (~sclrn) begin
                finish_out <= 1'b0;
            end
        end
    end

    always @ (posedge clock or negedge aclrn) begin
        if (~aclrn) begin
            finish_out_asserted <= 1'b0;
        end
        else begin
            if ( start ) begin
                finish_out_asserted <= 1'b0;
            end
            else if ( finish_out ) begin
                finish_out_asserted <= 1'b1;
            end
            if (~sclrn) begin
                finish_out_asserted <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
