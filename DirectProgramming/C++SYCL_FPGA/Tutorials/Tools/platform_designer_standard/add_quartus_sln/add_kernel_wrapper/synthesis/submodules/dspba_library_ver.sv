// Legal Notice: Copyright 2017 Intel Corporation.  All rights reserved.
// Your use of  Intel  Corporation's design tools,  logic functions and other
// software and tools,  and its AMPP  partner logic functions, and  any output
// files  any of the  foregoing  device programming or simulation files),  and
// any associated  documentation or information are expressly subject  to  the
// terms and conditions  of the Intel FPGA Software License Agreement,
// Intel  MegaCore  Function  License  Agreement, or other applicable license
// agreement,  including,  without limitation,  that your use  is for the sole
// purpose of  programming  logic  devices  manufactured by Intel and sold by
// Intel or its authorized  distributors.  Please  refer  to  the  applicable
// agreement for further details.


module dspba_delay_ver
#(
    parameter width = 8,
    parameter depth = 1,
    parameter reset_high = 1'b1,
    parameter reset_kind = "ASYNC",
    parameter phase = 0,
    parameter modulus = 1 
) (
    input clk,
    input aclr,
    input ena,
    input [width-1:0] xin,
    output [width-1:0] xout
);

    wire reset;
    reg [width-1:0] delays [depth-1:0];

    assign reset = aclr ^ reset_high;
    
    generate
        if (depth > 0)
        begin
            genvar i;
            for (i = 0; i < depth; ++i)
            begin : delay_block
                if ((reset_kind == "ASYNC") && (0 == (phase + i) % modulus))
                begin : async_reset
                always @ (posedge clk or negedge reset)
                    begin: a
                        if (!reset) begin
                            delays[i] <= 0;
                        end else begin
                            if (ena) begin
                                if (i > 0) begin
                                    delays[i] <= delays[i - 1];
                                end else begin
                                    delays[i] <= xin;
                                end
                            end
                        end
                    end
                end

                if ((reset_kind == "SYNC") && (0 == (phase + i) % modulus))
                begin : sync_reset
                always @ (posedge clk)
                    begin: a
                        if (!reset) begin
                            delays[i] <= 0;
                        end else begin
                            if (ena) begin
                                if (i > 0) begin
                                    delays[i] <= delays[i - 1];
                                end else begin
                                    delays[i] <= xin;
                                end
                            end
                        end
                    end
                end

                if ((reset_kind == "NONE") || (0 != (phase + i) % modulus))
                begin : no_reset
                always @ (posedge clk)
                    begin: a
                        if (ena) begin
                            if (i > 0) begin
                                delays[i] <= delays[i - 1];
                            end else begin
                                delays[i] <= xin;
                            end
                        end
                    end
                end
            end

            assign xout = delays[depth - 1];
        end else begin
            assign xout = xin;
        end
    endgenerate
    
endmodule

//------------------------------------------------------------------------------

module dspba_sync_reg_ver
#(
    parameter width1 = 8,
    parameter width2 = 8,
    parameter depth = 2,
    parameter pulse_multiplier = 1,
    parameter counter_width = 8,
    parameter init_value = 0,
    parameter reset1_high = 1'b1,
    parameter reset2_high = 1'b1,
    parameter reset_kind = "ASYNC",
    parameter implementation = "ASYNC" // ASYNC, SYNC, SYNC_LITE 
) (
    input clk1,
    input aclr1,
    input [0 : 0] ena,
    input [width1-1 : 0] xin,
    output [width1-1 : 0] xout,
    input clk2,
    input aclr2,
    output [width2-1 : 0] sxout
);
wire [width1-1 : 0] init_value_internal;

wire reset1;
wire reset2;

reg iclk_enable;
reg [width1-1 : 0] iclk_data;
(* altera_attribute = {"-name SYNCHRONIZER_IDENTIFICATION OFF"} *) reg [width2-1 : 0] oclk_data;

// For Synthesis this means: preserve this registers and do not merge any other flip-flops with synchronizer flip-flops 
// For TimeQuest this means: identify these flip-flops as synchronizer to enable automatic MTBF analysis
(* altera_attribute = {"-name ADV_NETLIST_OPT_ALLOWED NEVER_ALLOW; -name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS; -name DONT_MERGE_REGISTER ON; -name PRESERVE_REGISTER ON; -name FORCE_SYNCH_CLEAR ON"} *) reg [depth-1 : 0] sync_regs;

wire oclk_enable;

wire ena_internal;
reg [counter_width-1 : 0] counter;

assign init_value_internal = init_value;

assign reset1 = aclr1 ^ reset1_high;
assign reset2 = aclr2 ^ reset2_high;

generate
    if (pulse_multiplier == 1)
    begin: no_multiplication
        assign ena_internal = ena[0];
    end
endgenerate

generate 
    if (implementation == "ASYNC")
    begin: impl_async_counter
        if (pulse_multiplier > 1)
        begin: multiplu_ena_pulse
            if (reset_kind == "ASYNC")
            begin: async_reset
                always @ (posedge clk1 or negedge reset1)
                begin
                    if (reset1 == 1'b0) begin
                        counter <= 0;
                    end else begin
                        if (counter > 0) begin
                            if (counter == pulse_multiplier - 1) begin
                                counter <= 0;
                            end else begin
                                counter <= counter + 2'd1;
                            end
                        end else begin
                            if (ena[0] == 1'b1) begin
                                counter <= 1;
                            end
                        end
                    end
                end
            end
            if (reset_kind == "SYNC")
            begin: sync_reset
                always @ (posedge clk1)
                begin
                    if (reset1 == 1'b0) begin
                        counter <= 0;
                    end else begin
                        if (counter > 0) begin
                            if (counter == pulse_multiplier - 1) begin
                                counter <= 0;
                            end else begin
                                counter <= counter + 2'd1;
                            end
                        end else begin
                            if (ena[0] == 1'b1) begin
                                counter <= 1;
                            end
                        end
                    end
                end
            end
            if (reset_kind == "NONE")
            begin: no_reset
                always @ (posedge clk1)
                begin
                    if (counter > 0) begin
                        if (counter == pulse_multiplier - 1) begin
                            counter <= 0;
                        end else begin
                            counter <= counter + 2'd1;
                        end
                    end else begin
                        if (ena[0] == 1'b1) begin
                            counter <= 1;
                        end
                    end
                end
            end
            
            assign ena_internal = counter > 0 ? 1'b1 : ena[0];
        end
    end
endgenerate

assign oclk_enable = sync_regs[depth - 1];

generate
    if (reset_kind == "ASYNC")
    begin: iclk_async_reset 
        always @ (posedge clk1 or negedge reset1) 
        begin
           if (reset1 == 1'b0) begin
               iclk_data <= init_value_internal;
               iclk_enable <= 1'b0;
           end else begin
               iclk_enable <= ena_internal;
               if (ena[0] == 1'b1) begin
                   iclk_data <= xin;
               end
           end
        end
    end
    if (reset_kind == "SYNC")
    begin: iclk_sync_reset 
        always @ (posedge clk1) 
        begin
           if (reset1 == 1'b0) begin
               iclk_data <= init_value_internal;
               iclk_enable <= 1'b0;
           end else begin
               iclk_enable <= ena_internal;
               if (ena[0] == 1'b1) begin
                   iclk_data <= xin;
               end
           end
        end
    end
    if (reset_kind == "NONE")
    begin: iclk_no_reset 
        always @ (posedge clk1) 
        begin
           iclk_enable <= ena_internal;
           if (ena[0] == 1'b1) begin
               iclk_data <= xin;
           end
        end
    end
endgenerate

generate
    if (implementation == "ASYNC")
    begin: impl_async_sync_regs
        genvar i;
        for (i = 0; i < depth; ++i)
        begin: sync_regs_block
            if (reset_kind == "ASYNC") 
            begin: sync_reg_async_reset
                always @ (posedge clk2 or negedge reset2) begin
                    if (reset2 == 1'b0) begin
                        sync_regs[i] <= 1'b0;
                    end else begin
                        if (i > 0) begin
                            sync_regs[i] <= sync_regs[i - 1];
                        end else begin
                            sync_regs[i] <= iclk_enable;
                        end
                    end
                end
            end
            if (reset_kind == "SYNC") 
            begin: sync_reg_sync_reset
                always @ (posedge clk2) begin
                    if (reset2 == 1'b0) begin
                        sync_regs[i] <= 1'b0;
                    end else begin
                        if (i > 0) begin
                            sync_regs[i] <= sync_regs[i - 1];
                        end else begin
                            sync_regs[i] <= iclk_enable;
                        end
                    end
                end
            end
            if (reset_kind == "NONE") 
            begin: sync_reg_no_reset
                always @ (posedge clk2) begin
                    if (i > 0) begin
                        sync_regs[i] <= sync_regs[i - 1];
                    end else begin
                        sync_regs[i] <= iclk_enable;
                    end
                end
            end
        end
    end
endgenerate

generate
    if (implementation == "ASYNC")
    begin: impl_async_oclk_data
        if (reset_kind == "ASYNC")
        begin: oclk_async_reset
            always @ (posedge clk2 or negedge reset2)
            begin
                if (reset2 == 1'b0) begin
                    oclk_data <= init_value_internal[width2-1 : 0];
                end else begin
                    if (oclk_enable == 1'b1) begin
                        oclk_data <= iclk_data[width2-1 : 0];
                    end
                end
            end
        end
        if (reset_kind == "SYNC")
        begin: oclk_sync_reset
            always @ (posedge clk2)
            begin
                if (reset2 == 1'b0) begin
                    oclk_data <= init_value_internal[width2-1 : 0];
                end else begin
                    if (oclk_enable == 1'b1) begin
                        oclk_data <= iclk_data[width2-1 : 0];
                    end
                end
            end
        end
        if (reset_kind == "NONE")
        begin: oclk_no_reset
            always @ (posedge clk2)
            begin
                if (oclk_enable == 1'b1) begin
                    oclk_data <= iclk_data[width2-1 : 0];
                end
            end
        end
    end
    if (implementation == "SYNC")
    begin: impl_sync_oclk_data
        if (reset_kind == "ASYNC")
        begin: oclk_async_reset
            always @ (posedge clk2 or negedge reset2)
            begin
                if (reset2 == 1'b0) begin
                    oclk_data <= init_value_internal[width2-1 : 0];
                end else begin
                    oclk_data <= iclk_data[width2-1 : 0];
                end
            end
        end
        if (reset_kind == "SYNC")
        begin: oclk_sync_reset
            always @ (posedge clk2)
            begin
                if (reset2 == 1'b0) begin
                    oclk_data <= init_value_internal[width2-1 : 0];
                end else begin
                    oclk_data <= iclk_data[width2-1 : 0];
                end
            end
        end
        if (reset_kind == "NONE")
        begin: oclk_no_reset
            always @ (posedge clk2)
            begin
                oclk_data <= iclk_data[width2-1 : 0];
            end
        end
    end
    if (implementation == "SYNC_LITE")
    begin: impl_sync_lite_oclk_data
        assign oclk_data = iclk_data[width2-1 : 0];
    end
endgenerate

assign xout = iclk_data;
assign sxout = oclk_data;

endmodule

//------------------------------------------------------------------------------

module dspba_pipe
#(
    parameter num_bits   = 8,
    parameter num_stages = 0,
    parameter init_value = 1'bx
) (
    input  clk,
    input  [num_bits-1:0] d,
    output [num_bits-1:0] q
);

    generate
        if (num_stages > 0)
        begin
            reg [num_bits-1:0] stage_array[num_stages-1:0];

            genvar i;
            for (i = 0; i < num_stages; ++i)
            begin : g_pipe
                always @ (posedge clk) begin
                    if (i>0) begin
                        stage_array[i] <= stage_array[i-1];
                    end else begin
                        stage_array[i] <= d;
                    end
                end
            end

            assign q = stage_array[num_stages-1];

        end else begin
            assign q = d;
        end
    endgenerate

endmodule

//--------------------------------------------------------------------------

// The only purpose of this wrapper is so that we can apply timing constraints
// to dcfifo_mixed_widths using SDC_ENTITY_FILE qsf setting. It appears that 
// SDC_ENTITY_FILE cannot be applied directly to dcfifo_mixed_widths.
module dspba_dcfifo_mixed_widths
#(
    parameter lpm_width = 1,
    parameter lpm_width_r = lpm_width,
    parameter lpm_numwords = 2,
    parameter lpm_showahead = "OFF",
    parameter lpm_type = "dcfifo_mixed_widths",
    parameter lpm_hint = "USE_EAB=ON",
    parameter overflow_checking = "ON",
    parameter underflow_checking = "ON",
    parameter use_eab = "ON",
    parameter add_ram_output_register = "OFF",
    parameter intended_device_family = "Stratix",
    parameter lpm_widthu = 1,
    parameter lpm_widthu_r = lpm_widthu,
    parameter clocks_are_synchronized = "FALSE",
    parameter rdsync_delaypipe = 0,
    parameter wrsync_delaypipe = 0,
    parameter write_aclr_synch = "ON",
    parameter read_aclr_synch = "ON"
) (
    input rdreq,
    input wrclk,
    input rdclk,
    input aclr,
    input wrreq,
    input [lpm_width-1:0] data,
    output rdempty,
    output [lpm_width_r-1:0] q
);

   dcfifo_mixed_widths  dcfifo_mixed_widths_component (
                .data (data),
                .rdclk (rdclk),
                .rdreq (rdreq),
                .wrclk (wrclk),
                .wrreq (wrreq),
                .q (q),
                .rdempty (rdempty),
                .wrfull (),
                .aclr (aclr),
                .eccstatus (),
                .rdfull (),
                .rdusedw (),
                .wrempty (),
                .wrusedw ());
    defparam
        dcfifo_mixed_widths_component.lpm_width  = lpm_width,
        dcfifo_mixed_widths_component.lpm_width_r  = lpm_width_r,
        dcfifo_mixed_widths_component.lpm_numwords  = lpm_numwords,
        dcfifo_mixed_widths_component.lpm_showahead  = lpm_showahead,
        dcfifo_mixed_widths_component.lpm_type  = lpm_type,
        dcfifo_mixed_widths_component.lpm_hint  = lpm_hint,
        dcfifo_mixed_widths_component.overflow_checking  = overflow_checking,
        dcfifo_mixed_widths_component.underflow_checking  = underflow_checking,
        dcfifo_mixed_widths_component.use_eab  = use_eab,
        dcfifo_mixed_widths_component.add_ram_output_register  = add_ram_output_register,
        dcfifo_mixed_widths_component.intended_device_family  = intended_device_family,
        dcfifo_mixed_widths_component.lpm_widthu  = lpm_widthu,
        dcfifo_mixed_widths_component.lpm_widthu_r  = lpm_widthu_r,
        dcfifo_mixed_widths_component.clocks_are_synchronized  = clocks_are_synchronized,
        dcfifo_mixed_widths_component.rdsync_delaypipe  = rdsync_delaypipe,
        dcfifo_mixed_widths_component.wrsync_delaypipe  = wrsync_delaypipe,
        dcfifo_mixed_widths_component.write_aclr_synch  = write_aclr_synch,
        dcfifo_mixed_widths_component.read_aclr_synch  = read_aclr_synch;

    endmodule
