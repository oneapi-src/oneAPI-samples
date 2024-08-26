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

`default_nettype none

module acl_shift_register_no_reset #(
    parameter int unsigned WIDTH,
    parameter int unsigned STAGES
) (
    input  wire              clock,
    input  wire  [WIDTH-1:0] D,
    output logic [WIDTH-1:0] Q
);
    genvar g;
    generate
    if (STAGES == 0) begin : NO_STAGES
        assign Q = D;
    end
    else begin : GEN_STAGES
        logic [WIDTH-1:0] pipe [STAGES-1:0];
        always_ff @(posedge clock) begin
            pipe[0] <= D;
        end
        for (g=1; g<STAGES; g++) begin : GEN_PIPE
            always_ff @(posedge clock) begin
                pipe[g] <= pipe[g-1];
            end
        end
        assign Q = pipe[STAGES-1];
    end
    endgenerate
    
endmodule

`default_nettype wire
