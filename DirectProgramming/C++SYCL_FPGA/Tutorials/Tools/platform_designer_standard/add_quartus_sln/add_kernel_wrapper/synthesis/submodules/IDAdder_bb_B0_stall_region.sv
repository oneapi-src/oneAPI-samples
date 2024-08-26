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

// SystemVerilog created from bb_IDAdder_B0_stall_region
// Created for function/kernel IDAdder
// SystemVerilog created on Tue Aug 20 10:13:47 2024


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name AUTO_ROM_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module IDAdder_bb_B0_stall_region (
    input wire [0:0] in_avst_iowr_nb_acl_c_OutputPipeID_pipe_channel_almostfull,
    output wire [31:0] out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata,
    output wire [0:0] out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid,
    input wire [31:0] in_arg_a,
    input wire [31:0] in_arg_b,
    input wire [0:0] in_stall_in,
    output wire [0:0] out_stall_out,
    output wire [0:0] out_valid_out,
    input wire [0:0] in_valid_in,
    input wire clock,
    input wire resetn
    );

    wire [0:0] GND_q;
    wire [0:0] IDAdder_B0_merge_reg_out_stall_out;
    wire [0:0] IDAdder_B0_merge_reg_out_valid_out;
    wire [32:0] i_add_i_idadder_26_2gr_a;
    wire [32:0] i_add_i_idadder_26_2gr_b;
    logic [32:0] i_add_i_idadder_26_2gr_o;
    wire [32:0] i_add_i_idadder_26_2gr_q;
    wire [0:0] i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_almostfull;
    wire [0:0] i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_stall;
    wire [0:0] i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_valid;
    wire [31:0] i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata;
    wire [0:0] i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid;
    wire [0:0] i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_o_stall;
    wire [0:0] i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_o_valid;
    wire [31:0] i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_buffer_out;
    wire [0:0] i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_stall_out;
    wire [0:0] i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_valid_out;
    wire [31:0] i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_buffer_out;
    wire [0:0] i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_stall_out;
    wire [0:0] i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_valid_out;
    wire [31:0] bgTrunc_i_add_i_idadder_26_2gr_sel_x_b;
    wire [0:0] bubble_join_i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_q;
    wire [0:0] bubble_select_i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_b;
    wire [31:0] bubble_join_i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_q;
    wire [31:0] bubble_select_i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_b;
    wire [31:0] bubble_join_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_q;
    wire [31:0] bubble_select_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_b;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_wireValid;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_wireStall;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_StallValid;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_toReg0;
    reg [0:0] SE_out_IDAdder_B0_merge_reg_fromReg0;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_consumed0;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_toReg1;
    reg [0:0] SE_out_IDAdder_B0_merge_reg_fromReg1;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_consumed1;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_toReg2;
    reg [0:0] SE_out_IDAdder_B0_merge_reg_fromReg2;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_consumed2;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_or0;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_or1;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_backStall;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_V0;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_V1;
    wire [0:0] SE_out_IDAdder_B0_merge_reg_V2;
    wire [0:0] SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_wireValid;
    wire [0:0] SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_backStall;
    wire [0:0] SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_V0;
    wire [0:0] SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_wireValid;
    wire [0:0] SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_and0;
    wire [0:0] SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_and1;
    wire [0:0] SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_backStall;
    wire [0:0] SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_V0;
    wire [0:0] SE_stall_entry_wireValid;
    wire [0:0] SE_stall_entry_backStall;
    wire [0:0] SE_stall_entry_V0;


    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr(BLACKBOX,14)@1
    // in in_stall_in@20000000
    // out out_stall_out@20000000
    IDAdder_i_llvm_fpga_sync_buffer_i32_000001r thei_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr (
        .in_buffer_in(in_arg_b),
        .in_i_dependence(GND_q),
        .in_stall_in(SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_backStall),
        .in_valid_in(SE_out_IDAdder_B0_merge_reg_V2),
        .out_buffer_out(i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_buffer_out),
        .out_stall_out(i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_stall_out),
        .out_valid_out(i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_valid_out),
        .clock(clock),
        .resetn(resetn)
    );

    // i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr(BLACKBOX,13)@1
    // in in_stall_in@20000000
    // out out_stall_out@20000000
    IDAdder_i_llvm_fpga_sync_buffer_i32_000000r thei_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr (
        .in_buffer_in(in_arg_a),
        .in_i_dependence(GND_q),
        .in_stall_in(SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_backStall),
        .in_valid_in(SE_out_IDAdder_B0_merge_reg_V1),
        .out_buffer_out(i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_buffer_out),
        .out_stall_out(i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_stall_out),
        .out_valid_out(i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_valid_out),
        .clock(clock),
        .resetn(resetn)
    );

    // SE_stall_entry(STALLENABLE,60)
    // Valid signal propagation
    assign SE_stall_entry_V0 = SE_stall_entry_wireValid;
    // Backward Stall generation
    assign SE_stall_entry_backStall = IDAdder_B0_merge_reg_out_stall_out | ~ (SE_stall_entry_wireValid);
    // Computing multiple Valid(s)
    assign SE_stall_entry_wireValid = in_valid_in;

    // IDAdder_B0_merge_reg(BLACKBOX,2)@0
    // in in_stall_in@20000000
    // out out_data_out@1
    // out out_stall_out@20000000
    // out out_valid_out@1
    IDAdder_B0_merge_reg theIDAdder_B0_merge_reg (
        .in_data_in(GND_q),
        .in_stall_in(SE_out_IDAdder_B0_merge_reg_backStall),
        .in_valid_in(SE_stall_entry_V0),
        .out_data_out(),
        .out_stall_out(IDAdder_B0_merge_reg_out_stall_out),
        .out_valid_out(IDAdder_B0_merge_reg_out_valid_out),
        .clock(clock),
        .resetn(resetn)
    );

    // SE_out_IDAdder_B0_merge_reg(STALLENABLE,50)
    always_ff @ (posedge clock or negedge resetn)
    begin
        if (!resetn)
        begin
            SE_out_IDAdder_B0_merge_reg_fromReg0 <= '0;
            SE_out_IDAdder_B0_merge_reg_fromReg1 <= '0;
            SE_out_IDAdder_B0_merge_reg_fromReg2 <= '0;
        end
        else
        begin
            // Successor 0
            SE_out_IDAdder_B0_merge_reg_fromReg0 <= SE_out_IDAdder_B0_merge_reg_toReg0;
            // Successor 1
            SE_out_IDAdder_B0_merge_reg_fromReg1 <= SE_out_IDAdder_B0_merge_reg_toReg1;
            // Successor 2
            SE_out_IDAdder_B0_merge_reg_fromReg2 <= SE_out_IDAdder_B0_merge_reg_toReg2;
        end
    end
    // Input Stall processing
    assign SE_out_IDAdder_B0_merge_reg_consumed0 = (~ (i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_stall) & SE_out_IDAdder_B0_merge_reg_wireValid) | SE_out_IDAdder_B0_merge_reg_fromReg0;
    assign SE_out_IDAdder_B0_merge_reg_consumed1 = (~ (i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_stall_out) & SE_out_IDAdder_B0_merge_reg_wireValid) | SE_out_IDAdder_B0_merge_reg_fromReg1;
    assign SE_out_IDAdder_B0_merge_reg_consumed2 = (~ (i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_stall_out) & SE_out_IDAdder_B0_merge_reg_wireValid) | SE_out_IDAdder_B0_merge_reg_fromReg2;
    // Consuming
    assign SE_out_IDAdder_B0_merge_reg_StallValid = SE_out_IDAdder_B0_merge_reg_backStall & SE_out_IDAdder_B0_merge_reg_wireValid;
    assign SE_out_IDAdder_B0_merge_reg_toReg0 = SE_out_IDAdder_B0_merge_reg_StallValid & SE_out_IDAdder_B0_merge_reg_consumed0;
    assign SE_out_IDAdder_B0_merge_reg_toReg1 = SE_out_IDAdder_B0_merge_reg_StallValid & SE_out_IDAdder_B0_merge_reg_consumed1;
    assign SE_out_IDAdder_B0_merge_reg_toReg2 = SE_out_IDAdder_B0_merge_reg_StallValid & SE_out_IDAdder_B0_merge_reg_consumed2;
    // Backward Stall generation
    assign SE_out_IDAdder_B0_merge_reg_or0 = SE_out_IDAdder_B0_merge_reg_consumed0;
    assign SE_out_IDAdder_B0_merge_reg_or1 = SE_out_IDAdder_B0_merge_reg_consumed1 & SE_out_IDAdder_B0_merge_reg_or0;
    assign SE_out_IDAdder_B0_merge_reg_wireStall = ~ (SE_out_IDAdder_B0_merge_reg_consumed2 & SE_out_IDAdder_B0_merge_reg_or1);
    assign SE_out_IDAdder_B0_merge_reg_backStall = SE_out_IDAdder_B0_merge_reg_wireStall;
    // Valid signal propagation
    assign SE_out_IDAdder_B0_merge_reg_V0 = SE_out_IDAdder_B0_merge_reg_wireValid & ~ (SE_out_IDAdder_B0_merge_reg_fromReg0);
    assign SE_out_IDAdder_B0_merge_reg_V1 = SE_out_IDAdder_B0_merge_reg_wireValid & ~ (SE_out_IDAdder_B0_merge_reg_fromReg1);
    assign SE_out_IDAdder_B0_merge_reg_V2 = SE_out_IDAdder_B0_merge_reg_wireValid & ~ (SE_out_IDAdder_B0_merge_reg_fromReg2);
    // Computing multiple Valid(s)
    assign SE_out_IDAdder_B0_merge_reg_wireValid = IDAdder_B0_merge_reg_out_valid_out;

    // i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr(BLACKBOX,11)@1
    // in in_i_stall@20000000
    // out out_o_stall@20000000
    IDAdder_i_io_full_acl_c_outputpipeid000000r thei_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr (
        .in_avst_iowr_nb_acl_c_OutputPipeID_pipe_channel_almostfull(in_avst_iowr_nb_acl_c_OutputPipeID_pipe_channel_almostfull),
        .in_i_stall(SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_backStall),
        .in_i_valid(SE_out_IDAdder_B0_merge_reg_V0),
        .out_o_almostfull(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_almostfull),
        .out_o_stall(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_stall),
        .out_o_valid(i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_valid),
        .clock(clock),
        .resetn(resetn)
    );

    // bubble_join_i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr(BITJOIN,39)
    assign bubble_join_i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_q = i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_almostfull;

    // bubble_select_i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr(BITSELECT,40)
    assign bubble_select_i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_b = bubble_join_i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_q[0:0];

    // SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr(STALLENABLE,59)
    // Valid signal propagation
    assign SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_V0 = SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_wireValid;
    // Backward Stall generation
    assign SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_backStall = i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_o_stall | ~ (SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_wireValid);
    // Computing multiple Valid(s)
    assign SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_and0 = i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_valid_out;
    assign SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_and1 = i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_valid_out & SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_and0;
    assign SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_wireValid = i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_out_o_valid & SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_and1;

    // SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr(STALLENABLE,55)
    // Valid signal propagation
    assign SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_V0 = SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_wireValid;
    // Backward Stall generation
    assign SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_backStall = in_stall_in | ~ (SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_wireValid);
    // Computing multiple Valid(s)
    assign SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_wireValid = i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_o_valid;

    // bubble_join_i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr(BITJOIN,43)
    assign bubble_join_i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_q = i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_out_buffer_out;

    // bubble_select_i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr(BITSELECT,44)
    assign bubble_select_i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_b = bubble_join_i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_q[31:0];

    // bubble_join_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr(BITJOIN,46)
    assign bubble_join_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_q = i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_out_buffer_out;

    // bubble_select_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr(BITSELECT,47)
    assign bubble_select_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_b = bubble_join_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_q[31:0];

    // i_add_i_idadder_26_2gr(ADD,10)@1
    assign i_add_i_idadder_26_2gr_a = {1'b0, bubble_select_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_b};
    assign i_add_i_idadder_26_2gr_b = {1'b0, bubble_select_i_llvm_fpga_sync_buffer_i32_arg_a_sync_buffer_idadder_26_0gr_b};
    assign i_add_i_idadder_26_2gr_o = $unsigned(i_add_i_idadder_26_2gr_a) + $unsigned(i_add_i_idadder_26_2gr_b);
    assign i_add_i_idadder_26_2gr_q = i_add_i_idadder_26_2gr_o[32:0];

    // bgTrunc_i_add_i_idadder_26_2gr_sel_x(BITSELECT,33)@1
    assign bgTrunc_i_add_i_idadder_26_2gr_sel_x_b = i_add_i_idadder_26_2gr_q[31:0];

    // i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr(BLACKBOX,12)@1
    // in in_i_stall@20000000
    // out out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata@20000000
    // out out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid@20000000
    // out out_o_stall@20000000
    IDAdder_i_iowr_nb_acl_c_outputpipeid000000r thei_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr (
        .in_i_data(bgTrunc_i_add_i_idadder_26_2gr_sel_x_b),
        .in_i_stall(SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_backStall),
        .in_i_valid(SE_out_i_llvm_fpga_sync_buffer_i32_arg_b_sync_buffer_idadder_26_1gr_V0),
        .in_unnamed_IDAdder0(bubble_select_i_io_full_acl_c_outputpipeid_pipe_channel_unnamed_idadder0_idadder_26_3gr_b),
        .out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata(i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata),
        .out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid(i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid),
        .out_o_ack(),
        .out_o_stall(i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_o_stall),
        .out_o_valid(i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_o_valid),
        .clock(clock),
        .resetn(resetn)
    );

    // ext_sig_sync_out(GPOUT,9)
    assign out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata = i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifodata;
    assign out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid = i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_out_iowr_nb_acl_c_OutputPipeID_pipe_channel_o_fifovalid;

    // sync_out_13(GPOUT,30)@0
    assign out_stall_out = SE_stall_entry_backStall;

    // sync_out_14(GPOUT,31)@1
    assign out_valid_out = SE_out_i_iowr_nb_acl_c_outputpipeid_pipe_channel_unnamed_idadder1_idadder_26_4gr_V0;

endmodule
