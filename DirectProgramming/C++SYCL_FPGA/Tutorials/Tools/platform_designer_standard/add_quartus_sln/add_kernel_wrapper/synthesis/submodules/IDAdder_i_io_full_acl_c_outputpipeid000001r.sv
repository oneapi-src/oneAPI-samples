// ------------------------------------------------------------------------- 
// High Level Design Compiler for Intel(R) FPGAs Version 2024.2.1 (Release Build #a1b6f61c6e)
// 
// Legal Notice: Copyright 2024 Intel Corporation.  All rights reserved.
// Your use of  Intel Corporation's design tools,  logic functions and other
// software and  tools, and its AMPP partner logic functions, and any output
// files any  of the foregoing (including  device programming  or simulation
// files), and  any associated  documentation  or information  are expressly
// subject  to the terms and  conditions of the  Intel FPGA Software License
// Agreement, Intel MegaCore Function License Agreement, or other applicable
// license agreement,  including,  without limitation,  that your use is for
// the  sole  purpose of  programming  logic devices  manufactured by  Intel
// and  sold by Intel  or its authorized  distributors. Please refer  to the
// applicable agreement for further details.
// ---------------------------------------------------------------------------

// SystemVerilog created from i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_sr
// Created for function/kernel IDAdder
// SystemVerilog created on Tue Aug 20 10:13:47 2024


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name AUTO_ROM_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module IDAdder_i_io_full_acl_c_outputpipeid000001r (
    input wire [0:0] in_i_stall,
    output wire [0:0] out_o_stall,
    output wire [0:0] out_o_data,
    output wire [0:0] out_o_valid,
    input wire [0:0] in_i_data,
    input wire [0:0] in_i_valid,
    input wire clock,
    input wire resetn
    );

    wire [0:0] VCC_q;
    wire [0:0] combined_valid_q;
    wire [0:0] data_mux_s;
    reg [0:0] data_mux_q;
    wire [0:0] not_sr_valid_q;
    reg [0:0] sr_q;
    reg [0:0] sr_valid_q;
    wire [0:0] stall_and_valid_q;


    // combined_valid(LOGICAL,2)
    assign combined_valid_q = in_i_valid | sr_valid_q;

    // stall_and_valid(LOGICAL,7)
    assign stall_and_valid_q = in_i_stall & combined_valid_q;

    // sr_valid(REG,6)
    always_ff @ (posedge clock or negedge resetn)
    begin
        if (!resetn)
        begin
            sr_valid_q <= $unsigned(1'b0);
        end
        else
        begin
            sr_valid_q <= stall_and_valid_q;
        end
    end

    // sync_out_7(GPOUT,9)@20000000
    assign out_o_stall = sr_valid_q;

    // not_sr_valid(LOGICAL,4)
    assign not_sr_valid_q = ~ (sr_valid_q);

    // sr(REG,5)
    always_ff @ (posedge clock or negedge resetn)
    begin
        if (!resetn)
        begin
            sr_q <= $unsigned(1'b0);
        end
        else if (not_sr_valid_q == 1'b1)
        begin
            sr_q <= in_i_data;
        end
    end

    // VCC(CONSTANT,1)
    assign VCC_q = 1'b1;

    // data_mux(MUX,3)
    assign data_mux_s = sr_valid_q;
    always_comb 
    begin
        unique case (data_mux_s)
            1'b0 : data_mux_q = in_i_data;
            1'b1 : data_mux_q = sr_q;
            default : data_mux_q = 1'b0;
        endcase
    end

    // sync_out_8(GPOUT,10)@20000000
    assign out_o_data = data_mux_q;
    assign out_o_valid = combined_valid_q;

endmodule
