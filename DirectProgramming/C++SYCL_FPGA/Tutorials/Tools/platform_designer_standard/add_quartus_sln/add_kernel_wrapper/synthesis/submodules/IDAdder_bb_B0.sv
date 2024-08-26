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

// SystemVerilog created from bb_IDAdder_B0
// Created for function/kernel IDAdder
// SystemVerilog created on Tue Aug 20 10:13:47 2024


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name AUTO_ROM_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module IDAdder_bb_B0 (
    input wire [31:0] in_arg_a,
    input wire [31:0] in_arg_b,
    input wire [0:0] in_avst_iowr_nb_acl_c_OutputPipeID_pipe_channel_almostfull,
    input wire [0:0] in_stall_in_0,
    input wire [0:0] in_valid_in_0,
    output wire [31:0] out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata,
    output wire [0:0] out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid,
    output wire [0:0] out_stall_out_0,
    output wire [0:0] out_valid_out_0,
    input wire clock,
    input wire resetn
    );

    wire [0:0] IDAdder_B0_branch_out_stall_out;
    wire [0:0] IDAdder_B0_branch_out_valid_out_0;
    wire [0:0] IDAdder_B0_merge_out_stall_out_0;
    wire [0:0] IDAdder_B0_merge_out_valid_out;
    wire [31:0] bb_IDAdder_B0_stall_region_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata;
    wire [0:0] bb_IDAdder_B0_stall_region_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid;
    wire [0:0] bb_IDAdder_B0_stall_region_out_stall_out;
    wire [0:0] bb_IDAdder_B0_stall_region_out_valid_out;


    // IDAdder_B0_merge(BLACKBOX,3)
    IDAdder_B0_merge theIDAdder_B0_merge (
        .in_stall_in(bb_IDAdder_B0_stall_region_out_stall_out),
        .in_valid_in_0(in_valid_in_0),
        .out_stall_out_0(IDAdder_B0_merge_out_stall_out_0),
        .out_valid_out(IDAdder_B0_merge_out_valid_out),
        .clock(clock),
        .resetn(resetn)
    );

    // IDAdder_B0_branch(BLACKBOX,2)
    IDAdder_B0_branch theIDAdder_B0_branch (
        .in_stall_in_0(in_stall_in_0),
        .in_valid_in(bb_IDAdder_B0_stall_region_out_valid_out),
        .out_stall_out(IDAdder_B0_branch_out_stall_out),
        .out_valid_out_0(IDAdder_B0_branch_out_valid_out_0),
        .clock(clock),
        .resetn(resetn)
    );

    // bb_IDAdder_B0_stall_region(BLACKBOX,4)
    IDAdder_bb_B0_stall_region thebb_IDAdder_B0_stall_region (
        .in_arg_a(in_arg_a),
        .in_arg_b(in_arg_b),
        .in_avst_iowr_nb_acl_c_OutputPipeID_pipe_channel_almostfull(in_avst_iowr_nb_acl_c_OutputPipeID_pipe_channel_almostfull),
        .in_stall_in(IDAdder_B0_branch_out_stall_out),
        .in_valid_in(IDAdder_B0_merge_out_valid_out),
        .out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata(bb_IDAdder_B0_stall_region_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata),
        .out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid(bb_IDAdder_B0_stall_region_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid),
        .out_stall_out(bb_IDAdder_B0_stall_region_out_stall_out),
        .out_valid_out(bb_IDAdder_B0_stall_region_out_valid_out),
        .clock(clock),
        .resetn(resetn)
    );

    // out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata(GPOUT,10)
    assign out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata = bb_IDAdder_B0_stall_region_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata;

    // out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid(GPOUT,11)
    assign out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid = bb_IDAdder_B0_stall_region_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid;

    // out_stall_out_0(GPOUT,12)
    assign out_stall_out_0 = IDAdder_B0_merge_out_stall_out_0;

    // out_valid_out_0(GPOUT,13)
    assign out_valid_out_0 = IDAdder_B0_branch_out_valid_out_0;

endmodule
