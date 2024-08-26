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


/*
   This module pipelines each input signal and replicates the pipeline by the specified amounts.
   The copies of the pipelines are typically used to break up the fanout of the input signals.
   One common use-case for this block is to pipeline and fanout a synchronous reset, for performance.
*/

module acl_fanout_pipeline #(
   parameter   PIPE_DEPTH = 1,   // The number of pipeline stages. A value of 0 is valid and means the input will be passed straight to the output.
   parameter   NUM_COPIES = 1 ,  // The number of copies of the pipeline. Minimum value 1.
   parameter   WIDTH = 1         // The width of the input and output bus (ie. the number of unique inputs to fanout and pipeline). Minimum value 1.
)(
   input wire     clk,
   input wire     [WIDTH-1:0] in,
   output logic   [NUM_COPIES-1:0][WIDTH-1:0] out
);

   logic [WIDTH-1:0] pipe [NUM_COPIES][PIPE_DEPTH:1] /* synthesis dont_merge */;

   genvar j;
   generate
      if (PIPE_DEPTH == 0) begin
         for (j=0;j<NUM_COPIES;j++) begin : GEN_OUTPUT_ASSIGNMENT_PIPE_DEPTH_0
            assign out[j] = in;  // Pass the input straight through
         end
      end else begin
         always @(posedge clk) begin
            for (int k=0;k<NUM_COPIES;k++) begin      // For each copy
               pipe[k][1] <= in;                      // Assign the input to Stage-1 of the pipe
               for (int i=2;i<=PIPE_DEPTH;i++) begin  // Implement the rest of the pipe
                  pipe[k][i] <= pipe[k][i-1];
               end
            end
         end

         for (j=0;j<NUM_COPIES;j++) begin : GEN_OUTPUT_ASSIGNMENT_PIPE_DEPTH_GREATER_THAN_0 // For each copy, assign the pipe output to the output of this module
            assign out[j] = pipe[j][PIPE_DEPTH];
         end
      end
   endgenerate

endmodule
