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


/////////////////////////////////////////////////////////////////////////////////////
//                                                                                 //
//  acl_reset_handler                                                              //
//                                                                                 //
//  This block handles the generation of asynchronous and synchronous reset        //
//  signals for an OpenCL system.  Only one type of reset will be generated        //
//  (based on the ASYNC_RESET parameter), with the other hard-wired to be          //
//  inactive, so that any logic relying on the unused reset signal will be         //
//  optimized away.                                                                //
//                                                                                 //
//  The block takes an active-low reset input that is meant to be fed by a         //
//  global signal.  For use as an asynchronous reset, the input signal directly    //
//  (or through a synchronizer) feeds the asynchronous reset output.  For use      //
//  as a synchronous reset signal, the input reset is synchronized with proper     //
//  metastability hardening.                                                       //
//                                                                                 //
//                           *-------------------------------------------------*   //
//                           |              AVAILABLE FEATURES                 |   //
//  *-------------------------*--------------*----------------*-----------------*  //
//  |       WIRING MODE       | Synchronizer | Pulse Extender | Fanout Pipeline |  //
//  *-------------------------*--------------*----------------*-----------------*  //
//  | synchronous             | Optional     | Optional       | Optional        |  //
//  | async pass-through      | -            | -              | -               |  //
//  | async with synchronizer | Must Enable  | -              | -               |  //
//  *-------------------------*--------------*----------------*-----------------*  //
//                                                                                 //
/////////////////////////////////////////////////////////////////////////////////////

/*
Example usage of the reset handler:

module my_module #(
    parameter bit ASYNC_RESET = 1,          // how do the registers CONSUME reset: 1 means registers are reset asynchronously, 0 means registers are reset synchronously
    parameter bit SYNCHRONIZE_RESET = 0     // before consumption, do we SYNCHRONIZE the reset: 1 means use a synchronizer (reset arrived asynchronously), 0 means passthrough (reset was already synchronized)
    ...
) (
    input wire clk,
    input wire i_resetn,                    // this signal is assumed to be routed on a global network (although that is not strictlty necessary)
   ...
);

    // local parameters
    localparam NUM_RESET_COPIES = 3;                    // select this number to reduce the fanout of the synchronous reset signal within this module
    localparam RESET_PIPE_DEPTH = 4;                    // select this number to allow adequate registers on the reset path for retiming

    // reset related signals
    // we will write code that uses both of these signals, but remember that one of them will be hard-wired to '1' based on whether we select asynchronous or synchronous reset
    logic                        aclrn;                 // only one async reset signal, no special handling for fanout
    logic [NUM_RESET_COPIES-1:0] sclrn;                 // multiple copies of synchronous reset to reduce fanout
    logic                        resetn_synchronized;   // use this signal to prevent sub-blocks from requiring additional synchronizers

    // instantiate the reset handler
    acl_reset_handler #(
        .ASYNC_RESET            (ASYNC_RESET),          // select whether reset should be consumed asynchronously (1) or synchronously (0)
        .USE_SYNCHRONIZER       (SYNCHRONIZE_RESET),    // select whether to synchronize the reset BEFORE CONSUMPTION
        .SYNCHRONIZE_ACLRN      (SYNCHRONIZE_RESET),    // if ASYNC_RESET == 1 && USE_SYNCHRONIZER == 1, select whether o_alcrn should use the synchronized reset
        .PIPE_DEPTH             (RESET_PIPE_DEPTH),
        .NUM_COPIES             (NUM_RESET_COPIES)
    ) acl_reset_handler_inst (
        .clk                    (clk),
        .i_resetn               (i_resetn),
        .o_aclrn                (aclrn),
        .o_sclrn                (sclrn),
        .o_resetn_synchronized  (resetn_synchronized)
    );

    // sample always block showing use of both aclrn and sclrn
    always @(posedge clk or negedge aclrn) begin
        // code to use the async reset
        // remember, if ASYNC_RESET is set to 0, then aclrn is hard-wired to '1', and all this logic is
        // optimized away, so no ACLR ports are actually used in that case
        if (~aclrn) begin          
            // reset EVERY register with aclrn
            // if something is missing from this list that register will hold its value when aclrn == 0, whereas the desired behavior is typically aclrn has no effect
            myreg1 <= '0;
            myreg2 <= '0;
            ...
        end
        else begin
            myreg1 <= ...;
            myreg2 <= ...;

            // code the sync reset
            // Since this comes at the BOTTOM of the code, the assignments here override assignments above, thus
            // the synchronous reset takes precendence over all other assignments in this always block.
            // Recommended coding style is that this code should be an exact copy of the async code above, 
            // but optionally with select lines commented out, since not all signals may need a synchronous
            // reset.  It is a good practice to only reset those registers that REQUIRE a reset here.
            if (~scnlrn[0]) begin   // select which copy of sclrn to use, to optimize fanout of each copy
                // reset only SOME register with sclrn
                myreg1 <= '0;
                //myreg2 <= '0;     // leave this code here, but commented out, to show that this signal has intentionally NOT been reset with sclrn
            end
        end
    end

    // sample sub-module instantiation that has its own synchronizer
    my_submodule1 #(
        .ASYNC_RESET        (ASYNC_RESET),  // pass this parameter on to sub modules
        ...
    ) my_submodule1_inst (
        .clk                (clk),
        .i_resetn           (i_resetn),     // pass the input reset signal straight through
        ...
    );

    // sample sub-module instantiation that does not have its own synchronizer
    my_submodule2 #(
        .ASYNC_RESET        (ASYNC_RESET),  // module may or may not have this parameter
        .SYNCHRONIZE_RESET  (0),            // some modules may selectively allow adding a synchronizer locally or not 
                                            // this would be passed on to the USE_SYNCHRONIZER parameter in my_submodule2's acl_reset_handler
        ...
    ) my_submodule2_inst (
        .clk                (clk),
        .i_resetn           (resetn_synchronized),
        ...
    );

    ...

endmodule
*/

`default_nettype none

module acl_reset_handler #(
    // Configure port wiring:
    parameter ASYNC_RESET = 0,          // set to 1 to select asynchronous reset output, 0 to select synchronous reset output
    parameter SYNCHRONIZE_ACLRN = 0,    // set to 1 to cause the aclrn output to be fed through the synchronizer (only if enabled), 0 to have it fed directly by i_resetn (when ASYNC_RESET = 1)

    // Configure the synchronizer block:
    parameter USE_SYNCHRONIZER = 1,     // set to 1 to enable a clock domain crossing synchronizer, 0 to use i_resetn directly without a synchronizer

    // Configure the pulse extender block:
    parameter PULSE_EXTENSION = 0,      // prolongs the duration of the synchronized reset pulse by PULSE_EXTENSION cycles
                                        // Parameter is only respected when reset is synchronous (ASYNC_RESET = 0)
    // Configure the fanout block:
    parameter PIPE_DEPTH = 1,           // number of pipeline stages for synchronous reset outputs (pipeline stages are added AFTER the synchronizer)
                                        // A value of 0 is valid and means the input will be passed straight to the output after the synchronizer chain
    parameter NUM_COPIES = 1,           // number of copies of the synchronous reset output. Minimum value 1.
    parameter DUPLICATE  = 0            // this parameter is now deprecated, will remove up in a future revision, see Case:535246 -- was previously used for maxfan=1
)(
    input  wire                   clk,
    input  wire                   i_resetn, // this MUST be an active-low reset signal, NOTE that if this signal is left disconnected, the output reset signals will be stuck ASSERTED
    output logic                  o_aclrn,  // asynchronous reset output, equal to i_resetn (synchronized if SYNCHRONIZE_ACLRN=1) if ASYNC_RESET=1, hard wired to '1' otherwise
    output logic [NUM_COPIES-1:0] o_sclrn,  // multiple copies of synchronous reset output, with 'dont_merge' constraints applied to the registers that feed them to help with fanout
                                            // these signals will be hard-wired to '1' if ASYNC_RESET is 1
    output logic     o_resetn_synchronized  // signal to drive reset to local sub-modules (to prevent the need for multiple reset synchronizers within a local block)
                                            // if ASYNC_RESET = 1 and SYNCHRONIZE_ACLRN = 0, this is a copy of i_resetn, otherwise this is the reset signal after the synchronizer block
);

    //////////////////////////////////////////
    //                                      //
    //  First stage: synchronize the reset  //
    //                                      //
    //////////////////////////////////////////
    
    localparam SYNCHRONIZER_DEFAULT_DEPTH = 3;  // number of register stages used in sychronizer, default is 3 for S10 devices, must always be 2 or larger for all devices
    logic resetn_synchronized;
    
    generate
    if (USE_SYNCHRONIZER) begin : GEN_SYNCHRONIZER
        (* altera_attribute = {"-name ADV_NETLIST_OPT_ALLOWED NEVER_ALLOW; -name SYNCHRONIZER_IDENTIFICATION FORCED; -name DONT_MERGE_REGISTER ON; -name PRESERVE_REGISTER ON  "} *) reg synchronizer_head;
        (* altera_attribute = {"-name ADV_NETLIST_OPT_ALLOWED NEVER_ALLOW; -name DONT_MERGE_REGISTER ON; -name PRESERVE_REGISTER ON"} *) reg [SYNCHRONIZER_DEFAULT_DEPTH-2:0] synchronizer_body;
        
        // formerly din_s1 in acl_std_synchronizer_nocut with no metastability sim and rst_value == 0
        always_ff @(posedge clk or negedge i_resetn) begin
            if (~i_resetn) begin
                synchronizer_head <= 1'b0;
            end
            else begin
                synchronizer_head <= 1'b1;
            end
        end
        
        // formerly dreg in acl_std_synchronizer_nocut with rst_value == 0
        if (SYNCHRONIZER_DEFAULT_DEPTH < 3) begin : GEN_SHALLOW_SYNCHRONIZER
            always_ff @(posedge clk or negedge i_resetn) begin
                if (~i_resetn) begin
                    synchronizer_body <= '0;
                end
                else begin
                    synchronizer_body <= synchronizer_head;
                end
            end
        end
        else begin : GEN_DEEP_SYNCHRONIZER
            always_ff @(posedge clk or negedge i_resetn) begin
                if (~i_resetn) begin
                    synchronizer_body <= '0;
                end
                else begin
                    synchronizer_body <= {synchronizer_body[SYNCHRONIZER_DEFAULT_DEPTH-3:0], synchronizer_head};
                end
            end
        end
        
        // formerly dout in acl_std_synchronizer_nocut
        assign resetn_synchronized = synchronizer_body[SYNCHRONIZER_DEFAULT_DEPTH-2];
    end
    else begin : NO_SYNCHRONIZER
        assign resetn_synchronized = i_resetn;
    end
    endgenerate
    
    
    
    /////////////////////////////////////
    //                                 //
    //  Second stage: pulse extension  //
    //                                 //
    /////////////////////////////////////
    
    logic resetn_extended;
    
    generate
    if (PULSE_EXTENSION > 0 && ASYNC_RESET == 0) begin : GEN_PULSE_EXTENDER     // ignore pulse extender argument if reset type is asynchronous
        // POWER-UP NOTE:
        // This module has no "reset" for its own internal state, and will power up to a random state (PR). If the MSB is 0 on power-up, a spurious
        // reset pulse of unknown length will be triggered while the counter counts down to -1. This is not a problem because the "real" reset pulse
        // will follow. The resulting behavior will be correct whether the "real" pulse arrives during or after the spurious pulse.
        
        // count goes from PULSE_EXTENSION down to -1
        localparam WIDTH = $clog2(PULSE_EXTENSION+1) + 1;   // Number of bits needed to represent PULSE_EXTENSION, plus one extra MSB for rollover detection
        logic [WIDTH-1:0] count;
        
        // Peculiar coding style is used to work around synthesis arithmetic LUT inference issue - see case 459381.
        // Must express all combinational logic as happening BEFORE the arithmetic logic to guarantee LUT depth 1. 
        logic [WIDTH-1:0] count_mod;
        logic [WIDTH-1:0] decr;

        // Present synthesis with an unmistakable counter
        always_ff @(posedge clk) begin      //no reset
            count <= count_mod - decr;
        end

        // Express control logic as coming before the arithmetic operands
        always_comb begin
            if (!resetn_synchronized) begin // active low reset is active, set count to PULSE_EXTENSION
                count_mod = PULSE_EXTENSION;
                decr = '0;
            end
            else begin                      // reset is not active, count down
                if (!count[WIDTH-1]) begin  // MSB = 0, counter has not reached -1 yet, keep counting down
                    count_mod = count;
                    decr = 1'b1;
                end
                else begin                  // counter rolled over, stay at -1
                    count_mod = '1;
                    decr = '0;
                end
            end
        end
        
        // original code (for readability)
        // always_ff @(posedge clk) begin
        //     if (!resetn_synchronized) count <= PULSE_EXTENSION;
        //     else if (!count[WIDTH-1]) count <= count - 1;
        //     else count <= -1;
        // end
        
        assign resetn_extended = count[WIDTH-1];
    end
    else begin : NO_PULSE_EXTENDER
        assign resetn_extended = resetn_synchronized;
    end
    endgenerate
    
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                                                                                        //
    //  Third stage: reset distribution management - make multiple copies and add pipeline registers to each  //
    //                                                                                                        //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    logic [NUM_COPIES-1:0] resetn_fanout_pipeline;
    
    generate
    if (PIPE_DEPTH > 0) begin : GEN_RESET_PIPELINE
        logic [NUM_COPIES-1:0] pipe [PIPE_DEPTH-1:0] /* synthesis dont_merge */;
        always_ff @(posedge clk) begin  //no reset
            pipe[0] <= {NUM_COPIES{resetn_extended}};
            for (int i=1; i<PIPE_DEPTH; i++) begin : GEN_RANDOM_BLOCK_NAME_R59
                pipe[i] <= pipe[i-1];
            end
        end
        assign resetn_fanout_pipeline = pipe[PIPE_DEPTH-1];
    end
    else begin : NO_RESET_PIPELINE
        assign resetn_fanout_pipeline = {NUM_COPIES{resetn_extended}};
    end
    endgenerate
    
    
    
    //////////////////////////////////
    //                              //
    //  Finally, drive the outputs  //
    //                              //
    //////////////////////////////////
    
    generate
    if (ASYNC_RESET) begin : GEN_ASYNC_RESET
        assign o_aclrn = (SYNCHRONIZE_ACLRN) ? resetn_synchronized : i_resetn;  //use a sychronized reset if USE_SYNCHRONIZER==1 && SYNCHRONIZE_ACLRN==1
        assign o_sclrn = '1;                                //tie off
        assign o_resetn_synchronized = o_aclrn;             //choose the selected reset before fanout pipeline, pulse extension is not allowed on async reset so this is the same as after the synchronizer stage
    end
    else begin : GEN_SYNC_RESET
        assign o_aclrn = '1;                                //tie off
        assign o_sclrn = resetn_fanout_pipeline;            //this already has synchronize, pulse extend, and fanout pipeline integrated in
        assign o_resetn_synchronized = resetn_extended;     //choose the selected reset before fanout pipeline
    end
    endgenerate
    
endmodule

`default_nettype wire
